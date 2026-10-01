"""
A Agent / D3 + A5-law: 
  (D3) "先验物理 x 修正算子" 的可识别性：数据只约束它们的和 -> 存在精确零空间；
       加入 IB/速率惩罚 (对修正项) 后才可识别 —— 这直接指向郭玲组 2603.24948 / 2606.03469
       的 latent correction 架构的理论缺口。
  (A5-law) 有效维度预算的指数律：保留耗散率 gamma 的模态到视界 n 所需 IB 预算
       tau_req ~ e^{2 n gamma} —— 数值验证 log tau 对 gamma 的斜率是否为 2n。
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(11)

# ---------------- A5-law ----------------
def m_of_gamma(gamma, n, sigma2):
    gamma = np.asarray(gamma, float)
    out = np.where(gamma < 1e-12, n * sigma2,
                   sigma2 * (1 - np.exp(-2 * n * gamma)) / (1 - np.exp(-2 * gamma)))
    return out if out.ndim else float(out)


def optimal_c(mu, n, sigma2, tau):
    inv_a = m_of_gamma(-np.log(np.abs(mu)), n, sigma2) / np.abs(mu) ** (2 * n)
    order = np.argsort(inv_a); s = np.sort(inv_a); c = np.zeros_like(s); th = None
    for k in range(1, len(s) + 1):
        cand = (tau + s[:k].sum()) / k
        if (k == len(s) or cand <= s[k]) and cand > s[k - 1] - 1e-15:
            th = cand; c[:k] = cand - s[:k]; break
    out = np.empty_like(c); out[order] = c
    return out, th, inv_a


print("=" * 84)
print("[A5-law] 保留第 k 个模态所需 IB 预算 tau_req vs e^{2 n gamma_k}（sigma^2=1e-3）")
gammas = np.array([0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
mu_g = np.exp(-gammas)
sigma2 = 1e-3
for n_ in [10, 50, 100]:
    taus = []
    for k in range(1, len(gammas) + 1):
        lo, hi = 1e-12, 1e9
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            c_, _, _ = optimal_c(mu_g, n_, sigma2, mid)
            if (c_ > 1e-14).sum() >= k: hi = mid
            else: lo = mid
        taus.append(hi)
    taus = np.array(taus)
    pred = sigma2 * np.exp(2 * n_ * gammas[:len(taus)]) / (1 - np.exp(-2 * gammas[:len(taus)]))
    ratio = taus[1:] / pred[1:]
    loc = [(np.log(taus[k+1]) - np.log(taus[k])) / (gammas[k+1] - gammas[k])
           for k in range(len(taus) - 1)]
    print(f"  n={n_:4d}: tau_req/pred = {np.array2string(ratio, precision=2, max_line_width=200)}")
    print(f"           逐点局部斜率 dlog(tau)/dgamma = "
          f"{np.array2string(np.array(loc), precision=1, max_line_width=200)}  (预测 2n={2*n_})")
    print("           tau_req       =", np.array2string(taus, precision=3, max_line_width=200))

# ---------------- D3 ----------------
print()
print("=" * 84)
print("[D3] 先验算子 P(theta) + 修正 C(psi) 的可识别性（线性参数化, 真值 A* 已知）")
d = 8
E = [rng.normal(size=(d, d)) for _ in range(3)]          # 先验基 (3 维)
F = [rng.normal(size=(d, d)) for _ in range(3)]          # 修正基 (3 维)
# 让 F 与 E 有交叠：把 F 的一个方向做成 E 的线性组合 -> 制造不可识别方向
F[2] = 0.7 * E[0] + 0.3 * E[1]

theta_star = np.array([1.0, 0.5, -0.3])
psi_star = np.array([0.4, -0.2, 0.8])
U = rng.normal(size=(d, 12))                             # 12 个输入样本（矩阵形式，无需真算子）
A_true = sum(theta_star[i] * E[i] for i in range(3)) + sum(psi_star[j] * F[j] for j in range(3))
Y = A_true @ U

G = np.column_stack([(M @ U).reshape(-1) for M in (E + F)])      # 设计矩阵 (d*12) x 6
theta_true = np.concatenate([theta_star, psi_star])
resid = G @ theta_true - Y.reshape(-1)
print(f"  数据拟合残差 (真值处) = {np.linalg.norm(resid):.3e}  (应为 0)")
s = np.linalg.svd(G, compute_uv=False)
print(f"  设计矩阵奇异值 = {np.array2string(s, precision=3)}")
print(f"  => 最小奇异值/最大 = {s[-1]/s[0]:.3e}：存在近不可识别方向（修正基与先验基交叠）")

H0 = 2 * G.T @ G
w, V = np.linalg.eigh(H0)
print(f"  无惩罚时 Hessian 最小特征值 = {w[0]:.3e}（~0 => 不可识别）")
v0 = V[:, 0]
print(f"    零方向 (dtheta, dpsi) = {np.array2string(v0, precision=3)}")
print(f"    该方向先验部分={np.array2string(v0[:3], precision=3)}  修正部分={np.array2string(v0[3:], precision=3)}")
print("    含义：把 dtheta 从先验搬到修正（或反之）几乎不改变数据拟合 —— 架构本身不可识别。")

print()
print("  加入 IB/速率惩罚 lambda*||psi||^2 （相当于给修正项的信息预算）后：")
for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
    H = 2 * G.T @ G + lam * np.diag(np.r_[np.zeros(3), np.ones(3)])
    w2 = np.linalg.eigvalsh(H)
    # 条件数式指标：最小特征值 / 最大特征值
    print(f"    lambda={lam:<8}: min eig={w2[0]:.3e}  cond={w2[-1]/w2[0]:.3e}")

print()
print("  蒙特卡洛：无惩罚 vs 有惩罚 的参数恢复误差（噪声 sigma=1e-3, 50 次重复）")
def fit(lam, noise=1e-3):
    Yn = Y + rng.normal(scale=noise, size=Y.shape)
    yv = Yn.reshape(-1)
    R = lam * np.diag(np.r_[np.zeros(3), np.ones(3)])
    sol = np.linalg.solve(G.T @ G + R, G.T @ yv)
    return sol

for lam in [0.0, 1e-6, 1e-3]:
    errs = []
    for _ in range(50):
        sol = fit(lam)
        errs.append(sol - theta_true)
    errs = np.array(errs)
    print(f"    lambda={lam:<7}: RMSE(theta)={np.sqrt((errs[:, :3]**2).mean()):.3e}   "
          f"RMSE(psi)={np.sqrt((errs[:, 3:]**2).mean()):.3e}")
print()
print("  结论：数据残差对 theta 与 psi 的'交换'近不敏感（零空间），")
print("        惩罚/预算只能压制 psi 的方差、不能恢复被吸收进修正的先验误差方向；")
print("        可识别性必须来自结构性约束（修正类与先验误差类不相交），而非调 beta。")
