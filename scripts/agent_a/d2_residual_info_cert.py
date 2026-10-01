"""
A Agent / D2: 残差证书 -> 信息损失证书 (residual-certified information loss)

动机: 2026 年的 Koopman 文献已经能给出"算子残差证书"(ResDMD 类, 2606.29083 / 2603.15091:
主角度不变性诊断 + 多步误差界), 而 IB/Koopman 表示学习一侧 (2510.13025) 只处理精确算子下的
信息分配, 完全没有"算子估计误差如何吃掉时间相干信息"这条链路。本文件把两者接上。

定义 (n 步时间相干信息):
  I_n(K) = 1/2 log det( I + M_n(K)^{-1} K^n C K^{nT} ),
  M_n(K) = sum_{i=0}^{n-1} K^i Sigma K^{iT}
(线性高斯潜动力学 z_t = K z_{t-1} + eps_t, z_{t-n} ~ N(0,C), eps ~ N(0,Sigma))

命题 (一阶 + 范数界): 设 K_hat = K + E, A := I + M_n^{-1} K^n C K^{nT}, 则
  delta_I := I_n(K_hat) - I_n(K)
         = 1/2 tr( A^{-1} [ -M_n^{-1} dM_n M_n^{-1} K^n C K^{nT} + M_n^{-1} dP ] ) + O(||E||^2)
  |delta_I| <= 1/2 ||A^{-1}||_2 * || M_n^{-1} dP - M_n^{-1} dM_n M_n^{-1} K^n C K^{nT} ||_F  + O(||E||^2)
其中 dP = d(K^n) C K^{nT} + K^n C d(K^n)^T, d(K^n) = sum_{i+j=n-1} K^i E K^j,
      dM_n = sum_{i<n} [ d(K^i) Sigma K^{iT} + K^i Sigma d(K^i)^T ].
关键: 界只依赖 (E 的范数, K, C, Sigma, n)，因此可把数据驱动残差 rho >= ||E|| 直接转成
"信息损失上界"——这正是"MI 从被估计的量变成被认证的量"。
"""
import numpy as np

rng = np.random.default_rng(7)


def mats(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    P = Kn @ C @ Kn.T
    return Kn, M, P


def MI(K, C, Sig, n):
    _, M, P = mats(K, C, Sig, n)
    A = np.eye(K.shape[0]) + np.linalg.solve(M, P)
    s, logdet = np.linalg.slogdet(A)
    return 0.5 * logdet


def dM_dP(K, E, C, Sig, n):
    """dM_n 与 dP (一阶, 精确到 E 的一次项)"""
    d = K.shape[0]
    dpow = []                      # dpow[i] = d(K^i)
    dpow.append(np.zeros((d, d)))
    for i in range(1, n + 1):
        acc = np.zeros((d, d))
        for j in range(i):
            acc += np.linalg.matrix_power(K, j) @ E @ np.linalg.matrix_power(K, i - 1 - j)
        dpow.append(acc)
    dM = np.zeros((d, d))
    for i in range(n):
        Ki = np.linalg.matrix_power(K, i)
        dM += dpow[i] @ Sig @ Ki.T + Ki @ Sig @ dpow[i].T
    Kn = np.linalg.matrix_power(K, n)
    dP = dpow[n] @ C @ Kn.T + Kn @ C @ dpow[n].T
    return dM, dP


def first_order(K, E, C, Sig, n):
    d = K.shape[0]
    Kn, M, P = mats(K, C, Sig, n)
    Minv = np.linalg.inv(M)
    A = np.eye(d) + Minv @ P
    dM, dP = dM_dP(K, E, C, Sig, n)
    inner = Minv @ dP - Minv @ dM @ Minv @ P
    dI_fo = 0.5 * np.trace(np.linalg.inv(A) @ inner)
    bound = 0.5 * np.linalg.norm(np.linalg.inv(A), 2) * np.linalg.norm(inner, "fro")
    return dI_fo, bound


print("=" * 84)
print("[B1] 一阶预测 与 范数界 的正确性/紧度（随机收缩 K, d=6, 随机 E 尺度扫描, n=8）")
d, n, sigma2 = 6, 8, 1e-3
Sig = sigma2 * np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
mu = np.array([0.999, 0.995, 0.98, 0.9, 0.7, 0.4])
K = Q @ np.diag(mu) @ Q.T
C = np.eye(d)
I0 = MI(K, C, Sig, n)
print(f"  I_n(K) = {I0:.6f}")
print("   ||E||_F   exact dI     first-order   fo-ratio   bound      bound/exact")
for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    E = rng.normal(size=(d, d))
    E = E / np.linalg.norm(E, "fro") * eps
    exact = MI(K + E, C, Sig, n) - I0
    fo, bd = first_order(K, E, C, Sig, n)
    print(f"  {eps:8.1e}  {exact:+.3e}  {fo:+.3e}   {fo/exact if exact!=0 else float('nan'):8.4f}  "
          f"{bd:.3e}   {bd/abs(exact):8.2f}")

print()
print("[B2] 残差 -> 信息损失 的放大律：n 依赖（固定 ||E||_F=1e-3）")
print("   n     I_n(K)     exact |dI|    fo-pred      bound       bound/|dI|")
for n_ in [1, 2, 5, 10, 20, 40, 80]:
    E = rng.normal(size=(d, d))
    E = E / np.linalg.norm(E, "fro") * 1e-3
    I0_ = MI(K, C, Sig, n_)
    exact = MI(K + E, C, Sig, n_) - I0_
    fo, bd = first_order(K, E, C, Sig, n_)
    print(f"  {n_:4d}   {I0_:.5f}   {abs(exact):.3e}   {fo:+.3e}   {bd:.3e}   {bd/abs(exact):8.2f}")

print()
print("[B3] 谱半径依赖：放大常数 ~ 1/(1-|lambda_max|) ?（n=20, ||E||=1e-3, 单个近单位特征值）")
for lam in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    mu_ = np.array([lam, 0.5, 0.4, 0.3, 0.2, 0.1])
    K_ = Q @ np.diag(mu_) @ Q.T
    E = rng.normal(size=(d, d))
    E = E / np.linalg.norm(E, "fro") * 1e-3
    I0_ = MI(K_, C, Sig, 20)
    exact = abs(MI(K_ + E, C, Sig, 20) - I0_)
    fo, bd = first_order(K_, E, C, Sig, 20)
    print(f"  lam_max={lam:<8} |dI|={exact:.3e}  1/(1-lam)={1/(1-lam):8.1f}  "
          f"|dI|*(1-lam)={exact*(1-lam):.3e}   bound={bd:.3e}")

print()
print("[B4] 与'被保留的有效维度'耦合: 当 d_eff 崩塌时, 同样的算子残差 dI 更小吗?")
n_ = 20
for mu_ in [np.array([0.999, 0.995, 0.98, 0.9, 0.7, 0.4]),
            np.array([0.999, 0.9, 0.5, 0.3, 0.2, 0.1])]:
    K_ = Q @ np.diag(mu_) @ Q.T
    E = rng.normal(size=(d, d)); E = E / np.linalg.norm(E, "fro") * 1e-3
    I0_ = MI(K_, C, Sig, n_)
    exact = abs(MI(K_ + E, C, Sig, n_) - I0_)
    fo, bd = first_order(K_, E, C, Sig, n_)
    print(f"  mu={np.round(mu_,3)}  I_n={I0_:.5f}  |dI|={exact:.3e}  rel={exact/I0_*100:.2f}%")
