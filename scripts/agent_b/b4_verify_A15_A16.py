"""
B Agent / B-D4: 独立核查 A-CLAIM-15 (统计口径证书) 与 A-CLAIM-16 (统一律两常数)

对 A/d9_cert_tighten.py 的三个疑点：
  P1 代码里 D_wc = np.linalg.norm(Mh) * rho，而 np.linalg.norm(二维数组) 默认是 **谱范数**
     ||M||_2，不是注释里写的 ||M||_F。|<M,E>| <= ||M||_2 * ||E||_* (核范数) <= ||M||_2*||E||_F
     不成立（||E||_* 可以远大于 ||E||_F），所以 "最坏情况口径" 是否真是最坏情况需要
     **对抗方向构造** 来检验，而不是靠 40/40 覆盖。
  P2 bootstrap 对 **轨迹行** 重采样，数据是 AR 型序列 (z_{t+1}=K z_t+eps)，行级重采样破坏
     序列相关 -> 方差被低估 -> 覆盖率是伪 95%。对照 **块(轨迹)级** bootstrap。
  P3 rho 用 split-half ||K1-K2||_F/sqrt(2) 估计，它本身是噪声量，且衡量的是"两次拟合之差"
     而不是 "Khat - K*"；作为证书输入是否偏小？用真值 ||Khat-K||_F 对照。

对 A/d10_misspec_angle.py 的核心疑点（比"还没测 MLP"更致命）：
  P4 若 c_1, c_2 是"普适常数"，换环境维数 d 应该不变。但按最小二乘敏感性
        ||delta_psi|| ~ ||P_{v_min} vec(Delta U)|| / sigma_min(G),   sigma_min(G) ~ sin(alpha)
     残差在与 7 维列空间正交的 (d^2 - 7) 维补空间里近似各向同性，故它在近零奇异方向上的
     投影 ~ ||residual|| / sqrt(d^2) = ||residual|| / d。
     => **可否证预测**: c_1(d) ~ const / d。若实测 c_1 * d 为常数，则 A 的 "c_1=0.16"
        不是物理，而是他选的 d=10 的稀释因子；真正的设计量是 sin(alpha) 与 1/d 的比值。
  P5 A 的构造是 **等角** (q 个修正方向都在同一 alpha 上)，他的 alpha_min 是"平均角"。
     若把 1 个方向做成近退化 (alpha)、其余 2 个方向放在 60 度，误差应当只由那个
     最小主角度决定 (sigma_min 机制) 而不是平均角。检验：err*sin(alpha_min) 是否仍为常数，
     并与等角构造的常数比较。
"""
import numpy as np

rng = np.random.default_rng(20260930)


# ============================ Part A: A-CLAIM-15 ============================
def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    P = Kn @ C @ Kn.T
    A = np.eye(d) + np.linalg.solve(M, P)
    return 0.5 * np.linalg.slogdet(A)[1]


def deriv_matrix(K, C, Sig, n, h=1e-6):
    d = K.shape[0]
    Gm = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            Gm[i, j] = (mi_lin(K + E, C, Sig, n) - mi_lin(K - E, C, Sig, n)) / (2 * h)
    return Gm


def one_dataset(N, T, d, n, sigma2, K, Sig, Q):
    Zs, Zn = [], []
    for _ in range(N):
        z = rng.normal(size=d); traj = [z]
        for _ in range(T - 1):
            z = K @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
        X = np.array(traj)
        Zs.append(X[:-1]); Zn.append(X[1:])
    return np.vstack(Zs), np.vstack(Zn)


print("=" * 96)
print("[P1] 最坏情况口径的'范数约定'检查：||M||_2 * rho 是否是真最坏情况？")
d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d)
Qtot = np.linalg.qr(rng.normal(size=(d, d)))[0]
Ktrue = Qtot @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Qtot.T
C = np.eye(d)
I_true = mi_lin(Ktrue, C, Sig, n)
Mh = deriv_matrix(Ktrue, C, Sig, n)
n2, nf = np.linalg.norm(Mh, 2), np.linalg.norm(Mh, "fro")
print(f"  ||M||_2 = {n2:.4e}   ||M||_F = {nf:.4e}   比值 = {nf/n2:.3f}  (代码用的是前者 -> 界更小)")
# 对抗方向：最大化 <M,E> s.t. ||E||_F = rho  -> E_adv = rho * M/||M||_F
rho = 1e-2
Eadv = rho * Mh / nf
exact_adv = abs(mi_lin(Ktrue + Eadv, C, Sig, n) - I_true)
print(f"  取 rho={rho}: 对抗方向 <M,E>/||M||_F*rho = {np.sum(Mh*Eadv)/(nf*rho):.6f}")
print(f"    真实损失(对抗方向) = {exact_adv:.4e}")
print(f"    D_wc(谱范数口径)   = {n2*rho:.4e}   -> 覆盖? {exact_adv <= n2*rho}")
print(f"    D_wc(Frobenius口径)= {nf*rho:.4e}   -> 覆盖? {exact_adv <= nf*rho}")
print("    含义：'最坏情况'必须写 ||M||_F*rho（Cauchy-Schwarz 配对 Frobenius 残差），")
print("          代码里的 np.linalg.norm(Mh) 是谱范数，它不是对 ||E||_F<=rho 的合法上界。")

print()
print("=" * 96)
print("[P2/P3] 复现 A-CLAIM-15 + 块级 bootstrap + 真值残差对照")
print("  N x T | err中位 | D_row/err 覆盖 | D_blk/err 覆盖 | rho_hat/rho_true | 宽度比 blk/row")
for N, T in [(20, 50), (50, 100), (100, 200)]:
    trials, B = 25, 200
    err_l, row_l, blk_l, cov_row, cov_blk, rr = [], [], [], 0, 0, []
    for _ in range(trials):
        Z, Znext = one_dataset(N, T, d, n, sigma2, Ktrue, Sig, Qtot)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Ml = deriv_matrix(Khat, C, Sig, n)
        err = abs(mi_lin(Khat, C, Sig, n) - I_true)
        # split-half 残差估计 (A 的做法)
        idx = rng.permutation(len(Z)); hh = len(idx) // 2
        K1 = np.linalg.lstsq(Z[idx[:hh]], Znext[idx[:hh]], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[idx[hh:]], Znext[idx[hh:]], rcond=None)[0].T
        rho_hat = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
        rho_true = np.linalg.norm(Khat - Ktrue, "fro")
        # 行级 bootstrap
        srow = np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            srow[b] = abs(float(np.sum(Ml * (Kb - Khat))))
        # 块级 bootstrap：整条轨迹重采样
        sblk = np.empty(B)
        rows_per = T - 1
        for b in range(B):
            ib = rng.integers(0, N, N)
            pick = np.concatenate([np.arange(i * rows_per, (i + 1) * rows_per) for i in ib])
            Kb = np.linalg.lstsq(Z[pick], Znext[pick], rcond=None)[0].T
            sblk[b] = abs(float(np.sum(Ml * (Kb - Khat))))
        D_row, D_blk = np.quantile(srow, 0.95), np.quantile(sblk, 0.95)
        err_l.append(err); row_l.append(D_row / err); blk_l.append(D_blk / err)
        cov_row += err <= D_row; cov_blk += err <= D_blk
        rr.append(rho_hat / max(rho_true, 1e-30)); sblk_wide = D_blk / D_row
        sblk_wide = D_blk / D_row
    blk_l = np.array(blk_l); row_l = np.array(row_l)
    print(f"  {N:3d}x{T:3d} | {np.median(err_l):.3e} | {np.median(row_l):7.2f} {cov_row}/{trials} "
          f"| {np.median(blk_l):7.2f} {cov_blk}/{trials} | {np.median(rr):7.3f} | {np.median(blk_l/row_l):.3f}")
print("  判读：若 blk/row > 1 且 row 覆盖率 < 95%，则 A 的'统计口径 95% 覆盖'是行级重采样")
print("        破坏序列相关造成的 **欠覆盖**；正确的证书应基于块级(或轨迹级)重采样。")

print()
print("=" * 96)
print("[P3b] bootstrap 未建模的东西：一阶线性化本身的误差 + 估计偏差")
N, T = 50, 100
Z, Znext = one_dataset(N, T, d, n, sigma2, Ktrue, Sig, Qtot)
Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
Ml = deriv_matrix(Khat, C, Sig, n)
lin_pred = float(np.sum(Ml * (Khat - Ktrue)))
exact = mi_lin(Khat, C, Sig, n) - I_true
print(f"  一阶预测 <M, Khat-K*> = {lin_pred:+.4e}   真实 dI = {exact:+.4e}   比值 = {lin_pred/exact:.4f}")
print(f"  => 在 ||Khat-K*||_F={np.linalg.norm(Khat-Ktrue,'fro'):.3e} 的尺度上一阶是否够用")
# 有偏性：E[MI(Khat)] vs MI(K)
vals = []
for _ in range(60):
    Zi, Zni = one_dataset(N, T, d, n, sigma2, Ktrue, Sig, Qtot)
    Kh = np.linalg.lstsq(Zi, Zni, rcond=None)[0].T
    vals.append(mi_lin(Kh, C, Sig, n))
vals = np.array(vals)
print(f"  E[I_n(Khat)] = {vals.mean():.6f} +- {vals.std()/np.sqrt(60):.6f}   I_n(K*) = {I_true:.6f}"
      f"   bias = {vals.mean()-I_true:+.3e}")
print(f"  bootstrap 以 Khat 为中心，只能给出离散度，**给不出这个 bias**；若 bias 与 Delta_st 同量级，"
      f"则 95% 分位数不是有效区间端点。")

# ============================ Part B: A-CLAIM-16 ============================
def orth(A):
    return np.linalg.qr(A)[0]


def build_F_cols(QE, p, q, alphas):
    """让第 j 个修正方向与 span(E) 成 angle alphas[j]（等角时传入同一个值）"""
    QE = orth(QE[:, :p])
    D = QE.shape[0]
    cols = []
    for j in range(q):
        a = np.deg2rad(alphas[j])
        u = orth(QE @ orth(rng.normal(size=(p, 1))))[0]
        b_raw = rng.normal(size=(D, 1))
        b = orth(b_raw - QE @ (QE.T @ b_raw))[:, 0]
        cols.append(np.cos(a) * u + np.sin(a) * b)
    return np.column_stack(cols)


def run(dop, p, q, alphas, delta_norm, noise, n_samples=600, trials=20):
    errs = []
    for _ in range(trials):
        E = [rng.normal(size=(dop, dop)) / dop for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        FB = build_F_cols(QE, p, q, alphas)
        F = [FB[:, j].reshape(dop, dop) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Dh = rng.normal(size=(dop, dop)); Dh = Dh / np.linalg.norm(Dh, "fro") * delta_norm
        A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh
        U = rng.normal(size=(dop, n_samples))
        Y = A @ U + rng.normal(scale=noise, size=(dop, n_samples))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        C_hat = sum(sol[p + j] * F[j] for j in range(q))
        Cm = sum(psi[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - Cm, "fro"))
    return np.mean(errs)


print()
print("=" * 96)
print("[P4] 常数是否普适：等角 alpha=10deg, ||D||=0.05, p=4,q=3，扫描环境维数 d")
print("     d      err      c1 = err*sin(a)/||D||     c1 * d     c1 * d^2")
amp = np.sin(np.deg2rad(10.0))
for dop in [4, 6, 8, 10, 14, 20]:
    e = run(dop, 4, 3, [10.0] * 3, 0.05, 1e-4, n_samples=600, trials=20)
    c1 = e * amp / 0.05
    print(f"  {dop:4d}  {e:.4e}      {c1:.4e}        {c1*dop:.4f}    {c1*dop*dop:.3f}")
print("  预测（若 c1 ~ const/d）：第三列近似常数。")

print()
print("[P4b] 同样扫描纯噪声项 c2（D=0, sigma=1e-2）")
print("     d      err      c2 = err*sin(a)/sigma     c2 * d")
for dop in [4, 6, 8, 10, 14, 20]:
    e = run(dop, 4, 3, [10.0] * 3, 0.0, 1e-2, n_samples=600, trials=20)
    c2 = e * amp / 1e-2
    print(f"  {dop:4d}  {e:.4e}      {c2:.4e}        {c2*dop:.4f}")

print()
print("[P4c] 样本数 n_samples 扫描（d=10, alpha=10deg）：c1 是否依赖数据量？")
for ns in [50, 200, 600, 2000]:
    e = run(10, 4, 3, [10.0] * 3, 0.05, 1e-4, n_samples=ns, trials=20)
    print(f"  n_samples={ns:5d}: err={e:.4e}  c1={e*amp/0.05:.4e}")

print()
print("=" * 96)
print("[P5] 等角 vs 单一退化方向：误差由 alpha_min 还是平均角决定？(d=10,p=4,q=3)")
print("   配置                         err        err*sin(alpha_min)/||D||   err*sin(mean)/||D||")
cases = [
    ("全部 10deg (等角)",        [10.0, 10.0, 10.0]),
    ("全部 30deg",               [30.0, 30.0, 30.0]),
    ("1个10deg + 2个60deg",      [10.0, 60.0, 60.0]),
    ("2个10deg + 1个60deg",      [10.0, 10.0, 60.0]),
    ("1个5deg + 2个80deg",       [5.0, 80.0, 80.0]),
]
for name, al in cases:
    e = run(10, 4, 3, al, 0.05, 1e-4, n_samples=600, trials=20)
    amin, amean = min(al), np.mean(al)
    print(f"  {name:24s} {e:.4e}      {e*np.sin(np.deg2rad(amin))/0.05:.4e}"
          f"              {e*np.sin(np.deg2rad(amean))/0.05:.4e}")
print("  判读：若 'sin(alpha_min)' 列在五种配置下比 'sin(mean)' 列更接近常数，")
print("        则 A 的律应当写成 **最小主角度 (sigma_min 机制)**，他的等角构造无法区分两者。")
