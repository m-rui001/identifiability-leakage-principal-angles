"""
A Agent / D6 (S2-P0): 把"残差 -> 信息损失"从一阶估计升级为**可认证区间**，并做覆盖率实验。

关键想法：δI 是 E 的**线性泛函**：  δI(E) = <M, E>_F := sum_ij M_ij E_ij
  - M 可由 dI 的线性映射作用于基矩阵 E_ij 精确提取（无需任何估计）；
  - 于是 |δI| <= ||M||_F * ||E||_F 是**Cauchy–Schwarz 尖锐界**（比 §4.4 的
    链式范数界紧得多：后者把 M 本身的范数也放大了）。
  - 二阶余项用 Hessian 的算子范数上界估计： |R2| <= 0.5 * ||H||_2 * ||E||_F^2。

证书形式：  I_true ∈ [ I(Khat) - Delta , I(Khat) + Delta ],
             Delta = ||M||_F * rho_hat + 0.5 * ||H||_2 * rho_hat^2
其中 rho_hat 是**从数据估出的**残差上界（不能用真 K，否则是作弊）。

覆盖率实验（线性真值）：Khat 由最小二乘拟合，rho_hat 由 split-half bootstrap 给出。
"""
import numpy as np

rng = np.random.default_rng(20260930)


def MI(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    P = Kn @ C @ Kn.T
    A = np.eye(d) + np.linalg.solve(M, P)
    return 0.5 * np.linalg.slogdet(A)[1]


def deriv_matrix(K, C, Sig, n):
    """提取 δI(E) = <M,E> 的矩阵 M（对每个基矩阵做一次有限差分）"""
    d = K.shape[0]
    h = 1e-6
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            M[i, j] = (MI(K + E, C, Sig, n) - MI(K - E, C, Sig, n)) / (2 * h)
    return M


def hessian_norm(K, C, Sig, n, h=1e-4, n_dirs=60):
    """用方向二阶差分估计 Hessian 算子范数的下界近似（作为上界仍需保守化，见文中说明）"""
    d = K.shape[0]
    I0 = MI(K, C, Sig, n)
    M0 = deriv_matrix(K, C, Sig, n)
    worst = 0.0
    for _ in range(n_dirs):
        E = rng.normal(size=(d, d)); E /= np.linalg.norm(E)
        Ip = MI(K + h * E, C, Sig, n)
        Im = MI(K - h * E, C, Sig, n)
        r2 = (Ip + Im - 2 * I0) / h ** 2            # E^T H E
        worst = max(worst, abs(r2))
    return worst


def simulate_and_fit(K, Sig, N, T, d, rng):
    """生成 N 条长度 T 的轨迹，最小二乘拟合 Khat（EDMD 风格）"""
    Z, Znext = [], []
    for _ in range(N):
        z = rng.normal(size=d)
        traj = [z]
        for _ in range(T - 1):
            z = K @ z + rng.normal(scale=np.sqrt(np.diag(Sig)))
            traj.append(z)
        X = np.array(traj)
        Z.append(X[:-1]); Znext.append(X[1:])
    Z = np.vstack(Z); Znext = np.vstack(Znext)
    Khat, *_ = np.linalg.lstsq(Z, Znext, rcond=None)
    return Khat.T, Z, Znext


print("=" * 84)
print("[C1] 尖锐界 vs 朴素范数界：δI 的线性泛函结构")
d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
mu = np.array([0.999, 0.99, 0.95, 0.85, 0.6, 0.3])
K = Q @ np.diag(mu) @ Q.T
C = np.eye(d)
M = deriv_matrix(K, C, Sig, n)
I0 = MI(K, C, Sig, n)
print(f"  I_n(K)={I0:.6f}   ||M||_F={np.linalg.norm(M):.4f}")
print("   ||E||_F    exact dI      <M,E>       sharp=|M|*|E|   sharp/exact   naive/exact")
for eps in [1e-6, 1e-5, 1e-4, 1e-3]:
    E = rng.normal(size=(d, d)); E = E / np.linalg.norm(E) * eps
    exact = MI(K + E, C, Sig, n) - I0
    lin = float(np.sum(M * E))
    sharp = np.linalg.norm(M) * eps
    # 朴素界（§4.4 的链式范数界，重算一次做对照）
    Minv = np.linalg.inv(sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T
                             for i in range(n)))
    Kn = np.linalg.matrix_power(K, n)
    P = Kn @ C @ Kn.T
    A = np.eye(d) + Minv @ P
    dpow = [np.zeros((d, d))]
    for i in range(1, n + 1):
        acc = np.zeros((d, d))
        for j in range(i):
            acc += np.linalg.matrix_power(K, j) @ E @ np.linalg.matrix_power(K, i - 1 - j)
        dpow.append(acc)
    dMn = sum(dpow[i] @ Sig @ np.linalg.matrix_power(K, i).T
              + np.linalg.matrix_power(K, i) @ Sig @ dpow[i].T for i in range(n))
    dP = dpow[n] @ C @ Kn.T + Kn @ C @ dpow[n].T
    inner = Minv @ dP - Minv @ dMn @ Minv @ P
    naive = 0.5 * np.linalg.norm(np.linalg.inv(A), 2) * np.linalg.norm(inner, "fro")
    print(f"  {eps:8.1e}  {exact:+.3e}   {lin:+.3e}   {sharp:.3e}      {sharp/abs(exact):8.2f}"
          f"      {naive/abs(exact):8.2f}")

print()
print("[C2] 覆盖率实验：从数据估 rho_hat，区间是否覆盖真信息损失？（线性真值）")
print("   trials  N   T    rho_hat(median)  I_true   I_hat    宽度Δ     覆盖  宽度/|真-估|")
cov_all = []
for N, T in [(20, 50), (50, 100), (100, 200)]:
    cover, widths, gaps, rhos = 0, [], [], []
    for trial in range(60):
        Khat, Z, Znext = simulate_and_fit(K, Sig, N, T, d, rng)
        # ---- 完全数据驱动的 rho_hat：split-half bootstrap ----
        idx = rng.permutation(len(Z))
        h1, h2 = idx[: len(Z) // 2], idx[len(Z) // 2:]
        K1 = np.linalg.lstsq(Z[h1], Znext[h1], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[h2], Znext[h2], rcond=None)[0].T
        rho_hat = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)      # 两个半样本估计之差 -> 尺度
        Mhat = deriv_matrix(Khat, C, Sig, n)
        Hn = hessian_norm(Khat, C, Sig, n, n_dirs=8)
        Delta = np.linalg.norm(Mhat) * rho_hat + 0.5 * Hn * rho_hat ** 2
        I_true = MI(K, C, Sig, n)
        I_hat = MI(Khat, C, Sig, n)
        if abs(I_hat - I_true) <= Delta:
            cover += 1
        widths.append(Delta); gaps.append(abs(I_hat - I_true)); rhos.append(rho_hat)
        cov_all.append(abs(I_hat - I_true) <= Delta)
    w = np.array(widths); g = np.array(gaps) + 1e-12
    print(f"   {trial+1:6d}  {N:3d} {T:4d}   {np.median(rhos):.4e}       "
          f"{I_true:.4f}  {np.median([MI(Khat, C, Sig, n)]):.4f}"
          f"   {np.median(w):.3e}   {cover}/60   {np.median(w/g):6.2f}")
print(f"  总体覆盖率 = {np.mean(cov_all)*100:.1f}%  (target >= 95%)")

print()
print("[C3] 收敛性：rho_hat 随样本量下降，区间宽度 ~ O(rho)；检查宽度是否可实用")
print("   N*T     median rho_hat   median Δ      median |I_hat - I_true|   覆盖率")
for N, T in [(10, 30), (20, 60), (50, 120), (100, 300)]:
    rhos, Ds, Gs, cov = [], [], [], 0
    for trial in range(40):
        Khat, Z, Znext = simulate_and_fit(K, Sig, N, T, d, rng)
        idx = rng.permutation(len(Z)); h1, h2 = idx[: len(Z) // 2], idx[len(Z) // 2:]
        K1 = np.linalg.lstsq(Z[h1], Znext[h1], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[h2], Znext[h2], rcond=None)[0].T
        rho = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
        Mh = deriv_matrix(Khat, C, Sig, n)
        D = np.linalg.norm(Mh) * rho
        rhos.append(rho); Ds.append(D); Gs.append(abs(MI(Khat, C, Sig, n) - MI(K, C, Sig, n)))
        cov += abs(MI(Khat, C, Sig, n) - MI(K, C, Sig, n)) <= D
    print(f"   {N*T:6d}   {np.median(rhos):.3e}   {np.median(Ds):.3e}   "
          f"{np.median(Gs):.3e}            {cov}/40")
