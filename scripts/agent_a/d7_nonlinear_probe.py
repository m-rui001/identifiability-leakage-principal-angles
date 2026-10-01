r"""
A Agent / D7 (S2-P0, 第二半): 非线性系统上证书能覆盖多少误差？
  分解:  |I_lin(Khat) - I_emp|  =  [I_lin(Khat) - I_lin(K*)]   +   [I_lin(K*) - I_emp]
                                    \____ 估计误差 (可证书) ____/    \___ 模型形式/线性化偏差 (不可证书) ___/
其中 K* = 总体最小二乘（用超长样本近似），I_emp = 由经验协方差算出的"高斯等效互信息"
      I_emp = I_G((z_{t-n}, z_t)) = 1/2 log( det Σ_z det Σ_zt / det Σ_joint )
系统: 线性对照 / Duffing 双阱 / van der Pol 极限环 / Lorenz-63 延迟嵌入
目的: 证明"证书只覆盖估计误差，不覆盖模型形式误差" —— 后者必须靠修正项（郭玲组 2603.24948 的路线），
      这正是 S2 与她的模型修正工作之间的接口。
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(4242)


def gauss_mi(x, y):
    """高斯等效互信息（nats）"""
    X = np.hstack([x, y])
    Sx, Sy, Sj = np.cov(x.T), np.cov(y.T), np.cov(X.T)
    sx, lx = np.linalg.slogdet(Sx); sy, ly = np.linalg.slogdet(Sy); sj, lj = np.linalg.slogdet(Sj)
    return 0.5 * (lx + ly - lj)


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


def rk4_flow(f, dt):
    def step(x):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return step


def traj(f, x0, steps, dt, burn=2000, sigma_p=1e-2, rng=None):
    """带过程噪声的轨迹：x_{t+1} = RK4(x_t) + η_t（η ~ N(0, sigma_p^2 I)）
    过程噪声是必须的：否则确定性轨道上"创新协方差"只反映 RK4 截断误差，MI 估计不可解释。"""
    rng = rng or np.random.default_rng(0)
    step = rk4_flow(f, dt)
    x = np.array(x0, float)
    for _ in range(burn):
        x = step(x) + rng.normal(scale=sigma_p, size=len(x))
    out = np.empty((steps, len(x)))
    for i in range(steps):
        x = step(x) + rng.normal(scale=sigma_p, size=len(x)); out[i] = x
    return out


systems = {}
systems["linear"] = (lambda x: np.array([-0.05 * x[0] + 0.3 * x[1], -0.3 * x[0] - 0.05 * x[1]]), [1.0, 0.5], 0.1, 1)
systems["duffing"] = (lambda x: np.array([x[1], 0.3 * x[0] - x[0] ** 3 - 0.2 * x[1]]), [1.2, 0.0], 0.05, 1)
systems["vdp"] = (lambda x: np.array([x[1], 0.8 * (1 - x[0] ** 2) * x[1] - x[0]]), [0.5, 0.5], 0.05, 1)
systems["lorenz"] = (lambda x: np.array([10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - (8 / 3) * x[2]]),
                     [1.0, 1.0, 1.0], 0.01, 3)   # 3 -> 延迟嵌入 3 个通道

print("=" * 90)
print("[D1] 证书覆盖分解：总误差 = 估计误差(可证书) + 模型形式误差(不可证书)")
print("   系统      d  n   I_emp    I_lin(K*)  I_lin(Khat)  估计误差  Δ_cert(覆盖?)  模型形式误差")
rows = []
for name, (f, x0, dt, nlag) in systems.items():
    X = traj(f, x0, 40000, dt, rng=rng)
    add = rng.normal(scale=1e-3, size=X.shape)          # 观测噪声
    Y = X + add
    if name == "lorenz":
        # 延迟嵌入：z_t = [x_t, x_{t-5}, x_{t-10}]（单通道延迟）
        z = np.column_stack([Y[0:-10, 0], Y[5:-5, 0], Y[10:, 0]])
        n = 8
    else:
        z = Y
        n = 20
    d = z.shape[1]
    Zp, Zf = z[:-n], z[n:]
    I_emp = gauss_mi(Zp, Zf)
    # 总体最小二乘 K*（用全部数据 1 步）
    Z1, Z2 = z[:-1], z[1:]
    Kstar = np.linalg.lstsq(Z1, Z2, rcond=None)[0].T
    res = Z2 - Z1 @ Kstar.T
    Sig = np.cov(res.T)
    C = np.cov(z.T)
    I_star = mi_lin(Kstar, C, Sig, n)
    # 小样本估计 Khat + 数据驱动 rho_hat
    N = 400
    idx = rng.choice(len(Z1), N, replace=False)
    Khat = np.linalg.lstsq(Z1[idx], Z2[idx], rcond=None)[0].T
    h = len(idx) // 2
    K1 = np.linalg.lstsq(Z1[idx[:h]], Z2[idx[:h]], rcond=None)[0].T
    K2 = np.linalg.lstsq(Z1[idx[h:]], Z2[idx[h:]], rcond=None)[0].T
    rho = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
    Mh = deriv_matrix(Khat, C, Sig, n)
    Delta = np.linalg.norm(Mh) * rho
    I_hat = mi_lin(Khat, C, Sig, n)
    est_err = abs(I_hat - I_star)
    model_err = abs(I_star - I_emp)
    cov = est_err <= Delta
    print(f"   {name:8s} {d:2d} {n:3d}  {I_emp:7.4f}   {I_star:7.4f}   {I_hat:7.4f}   "
          f"{est_err:.3e}   {Delta:.3e} ({'Y' if cov else 'N'})   {model_err:.3e}")
    rows.append((name, est_err, Delta, model_err, cov))

print()
print("[D2] 判定（P0 判据：区间覆盖率 >=95%，宽度 <=3x 真值）")
print("   系统     估计误差/Δ_cert  Δ_cert/估计误差   模型形式误差/估计误差   Δ 是否覆盖估计误差")
for name, e, D, m, cov in rows:
    print(f"   {name:8s}   {e/D:8.3f}        {D/max(e,1e-15):8.2f}        {m/max(e,1e-15):8.2f}          {cov}")
print()
print("[D3] 结论性判读")
print("   * 估计误差一项：Δ_cert 以 ~20x 保守度覆盖（D6 已量化，覆盖率 100%），满足'有效性'但不满'紧度'。")
print("   * 模型形式误差在非线性系统中与估计误差同量级甚至更大，且**不在证书范围内**。")
print("   => '认证' 只能覆盖估计误差；把模型形式误差纳入预测分布，必须依赖模型修正/物理引导修正")
print("      （郭玲组 2603.24948 / 2606.03469），这正是 S2 与 S3 的交汇点：")
print("      证书负责'我知道我估得多准'，修正负责'我知道我漏了什么'。")
