"""
A Agent / D1 (v2): 线性高斯 Koopman 潜表示：信息-谱对应 / water-filling / 崩塌阈值 / 双区制信息视界
对照 2510.13025 (Information Shapes Koopman Representation, ICLR 2026)：
  - 他们给出"MI 最大化 -> water-filling"与定性"谱偏斜 -> 低秩崩塌"；
  - 本文补：精确留存判据、双区制视界的闭式、有效维度预算 d_eff(n,theta)。

模型 z_t = K z_{t-1} + eps_t (eps~N(0,Sigma))，z_{t-n}~N(0,C) 独立于噪声路径：
  I(z_{t-n};z_t) = 1/2 log det(I + M_n^{-1/2} K^n C K^{nT} M_n^{-1/2}),  M_n = sum_{i<n} K^i Sigma K^{iT}
对角化 (Sigma=sigma^2 I, K=diag(mu), C=diag(c))：
  a_j = |mu_j|^{2n}/m_j, m_j = sigma^2 (1-|mu_j|^{2n})/(1-|mu_j|^2)
  max I s.t. sum c_j = tau  =>  c_j = max(0, theta - 1/a_j)  [water-filling],  I* = 1/2 sum_{j in S} log(a_j theta)

精确留存判据（本文）： j 得到正方差 <=> r_j := m_j/|mu_j|^{2n} < theta
  <=> sigma^2 (e^{2 n gamma_j} - 1) < theta (1 - e^{-2 gamma_j}),  gamma_j = -log|mu_j|
双区制渐近（本文）：
  (i) 慢模态 n*gamma << 1:  判据 ~= n < theta/sigma^2   —— 共通信息视界，与 gamma 无关
  (ii) 快模态 n*gamma >> 1: 判据 ~= gamma < gamma* ~= log(theta/(2 gamma sigma^2))/(2n)
"""
import numpy as np
from scipy.optimize import minimize


def m_of_gamma(gamma, n, sigma2):
    """M_n 的对角元；gamma=0 (|mu|=1) 时取极限 n*sigma^2（扩散累积）"""
    gamma = np.asarray(gamma, dtype=float)
    out = np.where(
        gamma < 1e-12,
        n * sigma2,
        sigma2 * (1 - np.exp(-2 * n * gamma)) / (1 - np.exp(-2 * gamma)),
    )
    return out if out.ndim else float(out)


def optimal_c(mu, n, sigma2, tau):
    """精确解：排序法求 theta（无迭代根寻找，避免 bisect 失败）"""
    inv_a = m_of_gamma(-np.log(np.abs(mu)), n, sigma2) / np.abs(mu) ** (2 * n)
    order = np.argsort(inv_a)
    s = np.sort(inv_a)
    c = np.zeros_like(s)
    th = None
    for k in range(1, len(s) + 1):
        cand = (tau + s[:k].sum()) / k
        if (k == len(s) or cand <= s[k]) and cand > s[k - 1] - 1e-15:
            th = cand
            c[:k] = cand - s[:k]
            break
    assert th is not None
    out = np.empty_like(c)
    out[order] = c
    return out, th, inv_a


def mi_diag(mu, c, n, sigma2):
    a = np.abs(mu) ** (2 * n) / m_of_gamma(-np.log(np.abs(mu)), n, sigma2)
    return 0.5 * np.sum(np.log1p(a * c))


def mc_mi_1d(mu_j, c_j, n, sigma2, N=2_000_000, rng=None):
    """逐坐标独立：1D 高斯 MI = -1/2 log(1-rho^2)，rho 由协方差精确给出（用于交叉验证闭式）"""
    m = m_of_gamma(-np.log(np.abs(mu_j)), n, sigma2)
    v1 = c_j
    v2 = np.abs(mu_j) ** (2 * n) * c_j + m
    cov = np.abs(mu_j) ** n * c_j
    return -0.5 * np.log1p(-(cov ** 2) / (v1 * v2))


def mc_mi_empirical(mu_j, c_j, n, sigma2, N=400_000, rng=None):
    """真·蒙特卡洛：抽样估计 (x,y) 的高斯 MI 用 -1/2 log(1-rho_hat^2) 的样本版（kNN 太慢，这里用矩估计）"""
    rng = rng or np.random.default_rng(1)
    m = m_of_gamma(-np.log(np.abs(mu_j)), n, sigma2)
    x = rng.normal(0, np.sqrt(c_j), N)
    y = (np.abs(mu_j) ** n) * x + rng.normal(0, np.sqrt(m), N)
    rho = np.corrcoef(x, y)[0, 1]
    return -0.5 * np.log1p(-rho ** 2)


def zeta(gamma, n, sigma2, theta):
    return sigma2 * (np.exp(2 * n * gamma) - 1) - theta * (1 - np.exp(-2 * gamma))


def report(title):
    print("\n" + "=" * 80)
    print(title)


rng = np.random.default_rng(0)
sigma2, tau = 1e-3, 1.0

report("[A1] 闭式 MI  vs  蒙特卡洛抽样")
mu = np.array([0.999, 0.99, 0.95, 0.8, 0.5])
n = 20
c0 = np.full(len(mu), tau / len(mu))
cf = mi_diag(mu, c0, n, sigma2)
mc = sum(mc_mi_1d(mu[j], c0[j], n, sigma2) for j in range(len(mu)))
mce = sum(mc_mi_empirical(mu[j], c0[j], n, sigma2) for j in range(len(mu)))
print(f"  closed-form={cf:.6f}   analytic-per-coord={mc:.6f}   empirical-MC={mce:.6f}"
      f"   (empirical rel.err={abs(mce-cf)/cf*100:.3f}%)")

report("[A2] 闭式 water-filling  vs  数值优化 (SLSQP)")
for tau_ in [1.0, 0.1, 0.01]:
    c_star, th, inv_a = optimal_c(mu, n, sigma2, tau_)
    neg = lambda x: -mi_diag(mu, np.maximum(x, 1e-300), n, sigma2)
    r = minimize(neg, np.full(len(mu), tau_ / len(mu)),
                 constraints=[{"type": "ineq", "fun": lambda x: tau_ - x.sum()}],
                 bounds=[(0, tau_)] * len(mu), method="SLSQP",
                 options={"maxiter": 2000, "ftol": 1e-16})
    print(f"  tau={tau_:<6}: closed-form I*={mi_diag(mu, c_star, n, sigma2):.9f}  "
          f"SLSQP I*={-r.fun:.9f}  |dc|max={np.abs(c_star-r.x).max():.2e}")
    print(f"           c*={np.round(c_star,6)}  theta={th:.6f}")

report("[A3] 精确留存判据 vs 双区制渐近预测")
gammas = np.array([0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
mu_g = np.exp(-gammas)
print("   n     d_eff  theta      n*_slow=theta/sigma^2   exact-kept(gamma)")
for n_ in [1, 5, 20, 100, 500, 2000]:
    c_star, th, _ = optimal_c(mu_g, n_, sigma2, tau)
    kept = c_star > 1e-14
    nslow = th / sigma2
    exact = zeta(gammas, n_, sigma2, th) < 0
    print(f"  {n_:5d}  {kept.sum():4d}   {th:.6f}   {nslow:12.2f}            "
          f"{np.round(gammas[kept],4)}   exact==kept:{np.array_equal(exact,kept)}")

print()
print("  [A3b] 双区制验证：慢模态共通视界 n*≈theta/sigma^2  vs  快模态个体阈值 gamma*(n)")
for n_ in [20, 100, 500]:
    c_star, th, _ = optimal_c(np.exp(-gammas), n_, sigma2, tau)
    kept = c_star > 1e-14
    nslow = th / sigma2
    print(f"    n={n_}: theta/sigma^2={nslow:.1f}")
    for g, k in zip(gammas, kept):
        g_exact = zeta(np.array([g]), n_, sigma2, th)[0] < 0
        lo = np.log(th / sigma2) + np.log1p(-np.exp(-2 * g))   # 快模态 (n gamma >> 1)
        g_fast = (2 * n_ * g) < lo
        g_slow = n_ < nslow + 1e-12                            # 慢模态 (n gamma << 1)
        print(f"       gamma={g:<7} keep={str(bool(k)):<5} fast-pred={str(bool(g_fast)):<5} "
              f"slow-pred={str(bool(g_slow)):<5} n*gamma={n_*g:8.3f}")

report("[A4] 守恒模态 (|mu|=1) 的有限信息视界：I*(n) 何时塌到 0（单模态, tau=1）")
print("   单模态 |mu|=1: I* = 1/2 log(1 + tau/(n sigma^2))  (预测) vs 数值")
print("   n        I*(num)      I*(pred)      rel.err     I*=0.35nats 时 n=tau/sigma^2=?")
for n_ in [1, 10, 100, 1000, 10000]:
    c_star, th, _ = optimal_c(np.array([1.0]), n_, sigma2, tau)
    I = mi_diag(np.array([1.0]), c_star, n_, sigma2)
    pred = 0.5 * np.log1p(tau / (n_ * sigma2))
    print(f"  {n_:6d}   {I:.8f}   {pred:.8f}   {abs(I-pred)/max(I,1e-12)*100:.3e}%")
print(f"   => 信息视界 n_H = tau/sigma^2 = {tau/sigma2:.1f}（此处 I* 恰为 1/2 log 2 = {0.5*np.log(2):.6f} nats）")

report("[A5] 有效维度预算：给定 d_eff 目标，需要多少 IB 预算 tau（sigma^2=1e-3, n=100）")
n_ = 100
print("   target d_eff   required tau      theta")
for k in range(1, len(gammas) + 1):
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        c_, th_, _ = optimal_c(mu_g, n_, sigma2, mid)
        if (c_ > 1e-14).sum() >= k:
            hi = mid
        else:
            lo = mid
    c_, th_, _ = optimal_c(mu_g, n_, sigma2, hi)
    print(f"   {k:6d}        {hi:12.6g}   {th_:12.6g}")
