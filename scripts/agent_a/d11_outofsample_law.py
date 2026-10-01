"""
A Agent / D11 —— 回应 B §37 的两条批评（这两条都对，我接受）：
  B-批评-1: d10 的"预测"列用同一行实测反推常数 ⇒ 自我拟合，不是检验。本文件改为**真·样本外预测**，
            不含任何拟合常数：
                err_pred = ( ||P_F Δ_hidden||_F + sqrt(q/n)·σ ) / sin(alpha_min)
            其中 P_F 是到修正类 span(F) 的正交投影（直接用 QF 算，不看误差），sqrt(q/n)σ 是噪声臂
            （B 在 §37.2 给出的闭式 c2 ≈ sqrt(q/n)）。
  B-批评-2: d9 的覆盖率 36/40/40 在 T=40 下功效不足（SE≈2.2%，与 95% 差 2.3σ）⇒ 本轮把 T 提到 400，
            报告 Wilson 区间，并把措辞降到实测能支撑的强度。
"""
import numpy as np

rng = np.random.default_rng(60602)


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
    leak = np.linalg.norm(QF @ (QF.T @ Dh.reshape(-1)))          # ||P_F Δ||，含 Δ 本身是算子的归一化
    noise_arm = np.sqrt(q / n_samples) * noise
    pred = (leak + noise_arm) / np.sin(alpha)
    return err, pred, leak, noise_arm


print("=" * 96)
print("[H1] 真·样本外预测（无拟合常数）：err_pred = (||P_F Δ|| + sqrt(q/n) σ)/sin α")
print("     d    α    ||Δ||   σ        n     err(实测)    err_pred    比值    ||P_FΔ||   噪声臂")
rows = []
cases = []
for d in [6, 10, 14, 20]:
    cases.append((d, 90, 0.1, 0.0, 600))
for a in [5, 10, 20, 45, 60, 90]:
    cases.append((10, a, 0.1, 0.0, 600))
for dn in [0.02, 0.05, 0.1, 0.2]:
    cases.append((10, 45, dn, 0.0, 600))
for s in [0.0, 1e-3, 1e-2, 3e-2]:
    cases.append((10, 45, 0.05, s, 600))
for n in [150, 600, 2400, 9600]:
    cases.append((10, 45, 0.02, 1e-2, n))
for (d, a, dn, s, n) in cases:
    res = np.array([one_run(d, 4, 3, a, dn, s, n) for _ in range(12)])
    err, pred, leak, na = res.mean(axis=0)
    rows.append(err / pred)
    print(f"   {d:4d} {a:4d}  {dn:5.2f}  {s:<7} {n:6d}   {err:.4e}  {pred:.4e}  {err/pred:6.3f}   {leak:.3e}  {na:.3e}")
rows = np.array(rows)
print(f"\n   总览：比值中位数 {np.median(rows):.3f}，均值 {rows.mean():.3f}，区间 [{rows.min():.3f}, {rows.max():.3f}]")
print("   判据（B §37.5(i) 要求）：无拟合常数的预测，若比值集中在 1 附近（±30%），则律成立且不是重述。")

print()
print("[H2] 两臂交叉检验（B §37.5(ii)）：n* ≈ d²(σ/||Δ||)² 处两臂相等")
for d, dn, s in [(10, 0.02, 1e-2), (10, 0.05, 1e-2), (6, 0.05, 1e-2)]:
    nstar = d * d * (s / dn) ** 2
    print(f"   d={d}, ||Δ||={dn}, σ={s}: 预测 n* = {nstar:.1f}")
    for n in [max(50, int(nstar / 4)), max(50, int(nstar)), max(50, int(4 * nstar))]:
        err, pred, leak, na = np.array([one_run(d, 4, 3, 45, dn, s, n) for _ in range(12)]).mean(axis=0)
        print(f"      n={n:6d}: 实测={err:.4e}  预测={pred:.4e}  比值={err/pred:.3f}  (漏={leak:.3e}, 噪={na:.3e})")

# ---------------- Part 2: high-power coverage ----------------
print()
print("=" * 96)
print("[H3] 统计口径证书的高功效覆盖率检验（T=400，Wilson 区间）")


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


d, n, sigma2, B = 6, 10, 1e-3, 400
Sig = sigma2 * np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
K = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
C = np.eye(d)
I_true = mi_lin(K, C, Sig, n)
for N, T, trials in [(20, 50, 400), (100, 200, 200)]:
    cw = cs = 0
    wc_r, st_r, wide = [], [], []
    for trial in range(trials):
        Z, Znext = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = K @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Z.append(X[:-1]); Znext.append(X[1:])
        Z = np.vstack(Z); Znext = np.vstack(Znext)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Mh = deriv_matrix(Khat, C, Sig, n)
        idx = rng.permutation(len(Z)); h = len(idx) // 2
        K1 = np.linalg.lstsq(Z[idx[:h]], Znext[idx[:h]], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[idx[h:]], Znext[idx[h:]], rcond=None)[0].T
        rho = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
        D_wc = np.linalg.norm(Mh) * rho
        stats = np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            stats[b] = abs(float(np.sum(Mh * (Kb - Khat))))
        D_st = np.quantile(stats, 0.95)
        err = abs(mi_lin(Khat, C, Sig, n) - I_true)
        cw += err <= D_wc; cs += err <= D_st
        wc_r.append(D_wc / max(err, 1e-15)); st_r.append(D_st / max(err, 1e-15))
        wide.append(D_st / D_wc)
    lo1, hi1 = wilson(cw, trials); lo2, hi2 = wilson(cs, trials)
    print(f"   N×T={N}×{T}, trials={trials}:")
    print(f"     最坏情况: 覆盖 {cw}/{trials} = {cw/trials:.1%}  Wilson[{lo1:.1%},{hi1:.1%}]"
          f"   宽度/误差中位 {np.median(wc_r):.2f}")
    print(f"     统计口径: 覆盖 {cs}/{trials} = {cs/trials:.1%}  Wilson[{lo2:.1%},{hi2:.1%}]"
          f"   宽度/误差中位 {np.median(st_r):.2f}   收紧 {np.median(wide):.3f}×")
print()
print("判读：只有 Wilson 下界 >= 95% 才写'覆盖率达标'；否则写'覆盖率在本功效下与 95% 相容'。")
