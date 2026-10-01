"""
A Agent / D12 —— 两件事（都是被 B §37 逼出来的）：
 (1) [H4] 修正 §14.2 的合并方式：两臂不是相加，而是**平方相加**（理由：泄漏是定向偏差、噪声是随机项，
     二者独立 ⇒ E||bias+noise||^2 = ||bias||^2 + E||noise||^2）。检验 err ≈ sqrt(||P_FΔ||² + (√(q/n)σ)²)/sin α。
 (2) [H5] 覆盖率的多分位检验：若 95% 分位欠覆盖，则提高分位（97.5% / 99%）能否把覆盖率补回 ≥95%，
     代价是宽度增加多少 —— 这就是 K1 的可操作配方。
"""
import numpy as np

rng = np.random.default_rng(60603)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha, d):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def one_run(d, p, q, alpha_deg, dn, noise, n_samples):
    alpha = np.deg2rad(alpha_deg)
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    FB = build_F_at_angle(QE, p, q, alpha, d)
    QF = orth(FB)
    F = [FB[:, j].reshape(d, d) for j in range(q)]
    theta = rng.normal(size=p); psi = rng.normal(size=q)
    Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * dn
    A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh
    U = rng.normal(size=(d, n_samples))
    Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
    sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
    C_hat = sum(sol[p + j] * F[j] for j in range(q))
    C = sum(psi[j] * F[j] for j in range(q))
    err = np.linalg.norm(C_hat - C, "fro")
    leak = np.linalg.norm(QF @ (QF.T @ Dh.reshape(-1)))
    na = np.sqrt(q / n_samples) * noise
    return err, leak, na


print("=" * 100)
print("[H4] 线性相加 vs 平方相加（同一批抽样，无拟合常数）")
print("     d    α    ||Δ||   σ       n      实测      lin预测   比值   quad预测   比值")
rc = {"lin": [], "quad": []}
cases = ([(d, 90, 0.1, 0.0, 600) for d in [6, 10, 14, 20]]
         + [(10, a, 0.1, 0.0, 600) for a in [5, 10, 20, 45, 60, 90]]
         + [(10, 45, dn, 0.0, 600) for dn in [0.02, 0.05, 0.1, 0.2]]
         + [(10, 45, 0.05, s, 600) for s in [0.0, 1e-3, 1e-2, 3e-2]]
         + [(10, 45, 0.02, 1e-2, n) for n in [150, 600, 2400, 9600]]
         + [(10, 45, 0.05, 3e-2, n) for n in [150, 600, 2400]]
         + [(10, 20, 0.05, 1e-2, 600), (14, 30, 0.1, 3e-2, 300), (6, 60, 0.02, 1e-2, 1200)])
for (d, a, dn, s, n) in cases:
    res = np.array([one_run(d, 4, 3, a, dn, s, n) for _ in range(10)])
    err, leak, na = res.mean(axis=0)
    sa = np.sin(np.deg2rad(a))
    lin = (leak + na) / sa
    quad = np.sqrt(leak ** 2 + na ** 2) / sa
    rc["lin"].append(err / lin); rc["quad"].append(err / quad)
    print(f"   {d:4d} {a:4d}  {dn:5.2f}  {s:<6} {n:6d}  {err:.3e}  {lin:.3e}  {err/lin:5.3f}  {quad:.3e}  {err/quad:5.3f}")
for k in rc:
    v = np.array(rc[k])
    print(f"   {k:5s}: 中位 {np.median(v):.3f}  均值 {v.mean():.3f}  区间 [{v.min():.3f},{v.max():.3f}]  "
          f"相对散度 std/mean {v.std()/v.mean():.1%}")

# ---------------- H5: 多分位覆盖率 ----------------
print()
print("=" * 100)
print("[H5] 覆盖率 vs 分位水平（同一批自助样本，改变分位几乎免费）")


def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    return 0.5 * np.linalg.slogdet(np.eye(d) + np.linalg.solve(M, Kn @ C @ Kn.T))[1]


def deriv_matrix(K, C, Sig, n, h=1e-6):
    d = K.shape[0]
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            M[i, j] = (mi_lin(K + E, C, Sig, n) - mi_lin(K - E, C, Sig, n)) / (2 * h)
    return M


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
K = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
C = np.eye(d)
I_true = mi_lin(K, C, Sig, n)
levels = [0.90, 0.95, 0.975, 0.99]
for N, T, trials, B in [(20, 50, 400, 400), (100, 200, 250, 400)]:
    cov = {l: 0 for l in levels}
    wide = {l: [] for l in levels}
    st_err = []
    for _ in range(trials):
        Z, Znext = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = K @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Z.append(X[:-1]); Znext.append(X[1:])
        Z = np.vstack(Z); Znext = np.vstack(Znext)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Mh = deriv_matrix(Khat, C, Sig, n)
        stats = np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            stats[b] = abs(float(np.sum(Mh * (Kb - Khat))))
        err = abs(mi_lin(Khat, C, Sig, n) - I_true)
        st_err.append(err)
        for l in levels:
            D = np.quantile(stats, l)
            cov[l] += err <= D
            wide[l].append(D / max(err, 1e-15))
    print(f"   N×T={N}×{T}, trials={trials}, B={B}（真误差中位 {np.median(st_err):.2e}）")
    for l in levels:
        lo, hi = wilson(cov[l], trials)
        ok = "**达标**" if lo >= 0.95 else "未达标"
        print(f"      分位 {l:.3f}: 覆盖 {cov[l]}/{trials} = {cov[l]/trials:6.1%}  Wilson[{lo:.1%},{hi:.1%}] {ok}"
              f"   宽度/误差中位 {np.median(wide[l]):5.2f}")
print()
print("判读：若把分位从 0.95 提到 0.975/0.99 后 Wilson 下界越过 95%，则 K1 的配方是"
      "'统计口径 + 分位膨胀'，代价是宽度从 ~3.1 涨到 ~?.（这比'加二阶余项'便宜得多，且可证。）")
