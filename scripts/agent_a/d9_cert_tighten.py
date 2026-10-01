"""
A Agent / D9 (S2-K1): 收紧证书 —— 最坏情况界 vs 自助法(bootstrap)统计界
  §12.1 的最坏情况证书 Δ_wc = ||M||_F * rho_hat 是"对抗性方向"的界，实测保守 17–21×。
  收紧思路（不牺牲有效性的前提下提供两种口径）：
    (A) 最坏情况（100% 覆盖）：  Δ_wc = ||M||_F * rho_hat
    (B) 统计口径（覆盖率 ~1-α）：Δ_st = |⟨M, Ê_boot - K̂⟩| 的 (1-α) 分位数（对数据行做 bootstrap）
  判据：K1 = "在 VdP/Lorenz 上 Δ/估计误差 <= 3 且覆盖率 >= 95%"。
"""
import numpy as np

rng = np.random.default_rng(20261001)


def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    P = Kn @ C @ Kn.T
    A = np.eye(d) + np.linalg.solve(M, P)
    return 0.5 * np.linalg.slogdet(A)[1]


def deriv_matrix(K, C, Sig, n, h=1e-6):
    d = K.shape[0]
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            M[i, j] = (mi_lin(K + E, C, Sig, n) - mi_lin(K - E, C, Sig, n)) / (2 * h)
    return M


print("=" * 92)
print("[F1] 最坏情况界 vs 自助法统计界（线性真值, d=6, n=10, B=300）")
print("   trials  N   T   Δ_wc/估计误差  Δ_st(95%)/估计误差  覆盖率(wc)  覆盖率(st)  宽度比 st/wc")
for N, T in [(20, 50), (50, 100), (100, 200)]:
    d, n, sigma2 = 6, 10, 1e-3
    Sig = sigma2 * np.eye(d)
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    K = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
    C = np.eye(d)
    I_true = mi_lin(K, C, Sig, n)
    wc_ratio, st_ratio, cov_wc, cov_st, wide = [], [], 0, 0, []
    for trial in range(40):
        Z, Znext = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = K @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Z.append(X[:-1]); Znext.append(X[1:])
        Z = np.vstack(Z); Znext = np.vstack(Znext)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Mh = deriv_matrix(Khat, C, Sig, n)
        # 最坏情况
        idx = rng.permutation(len(Z)); h = len(idx) // 2
        K1 = np.linalg.lstsq(Z[idx[:h]], Znext[idx[:h]], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[idx[h:]], Znext[idx[h:]], rcond=None)[0].T
        rho = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
        D_wc = np.linalg.norm(Mh) * rho
        # 自助法：对数据行重采样，得到统计量的分布
        B = 300
        stats = np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            stats[b] = abs(float(np.sum(Mh * (Kb - Khat))))
        D_st = np.quantile(stats, 0.95)
        err = abs(mi_lin(Khat, C, Sig, n) - I_true)
        if err <= D_wc: cov_wc += 1
        if err <= D_st: cov_st += 1
        wc_ratio.append(D_wc / max(err, 1e-15)); st_ratio.append(D_st / max(err, 1e-15))
        wide.append(D_st / D_wc)
    print(f"   {40:6d} {N:3d} {T:4d}      {np.median(wc_ratio):8.2f}        {np.median(st_ratio):8.2f}"
          f"          {cov_wc}/40       {cov_st}/40      {np.median(wide):.3f}")
print()
print("[F2] 判读")
print("  * 若 Δ_st 覆盖率 ≈95%（不显著低于 95%）且 Δ_st/Δ_wc << 1，则'统计口径'是可用的收紧方式；")
print("    它把'最坏情况'换成'给定覆盖率'，这是工程上正确的取舍（与 2026 conformal 系列同一哲学）。")
print("  * 若 Δ_st 覆盖率明显 <95%（欠覆盖），说明一阶线性化在估计误差尺度上失真，需要二阶余项。")
