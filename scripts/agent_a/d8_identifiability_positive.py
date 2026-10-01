"""
A Agent / D8 (S3-K2): 修正项可识别性的**正面结果** —— 主角度判据
  设定：真算子  A* = P(θ*) + C(ψ*)，  先验类 span(E)（dim p），修正类 span(F)（dim q）
  数据只约束 P(θ)+C(ψ)。问：什么时候 (θ, ψ) 可识别？恢复误差是多少？
  预测（本文件验证）：恢复误差 ~ 真修正项在**先验类**上的投影范数
        err_ψ ≈ || Π_{span(E)} C* ||_F   （外加噪声项）
      ⇒ 可识别性由 span(E) 与 span(F) 的**主角度**控制：两者正交 ⇒ 完全可识别。
  与 2603.15091 的接口：他们用主角度度量 Koopman 子空间的不变性；这里用主角度度量
  "先验类 vs 修正类"的可分离性 —— 同一工具的第二次使用。
"""
import numpy as np
from scipy.linalg import subspace_angles

rng = np.random.default_rng(77)
d = 8


def rand_op():
    return rng.normal(size=(d, d)) / np.sqrt(d)


def flatten_basis(B):
    return np.column_stack([b.reshape(-1) for b in B])


def run_case(p, q, overlap_angle_deg, noise, n_samples=400, trials=40):
    """overlap_angle_deg: 通过旋转让 F 的一个方向与 span(E) 成给定主角度"""
    errs, errs_pred, errs_theta = [], [], []
    for _ in range(trials):
        E = [rand_op() for _ in range(p)]
        GE = flatten_basis(E)
        QE, _ = np.linalg.qr(GE)
        F = [rand_op() for _ in range(q)]
        # 让 F 的最后一位与 span(E) 成指定角度
        v = F[-1].reshape(-1)
        v_perp = v - QE @ (QE.T @ v)
        if np.linalg.norm(v_perp) < 1e-12:
            v_perp = rng.normal(size=v.shape)
            v_perp -= QE @ (QE.T @ v_perp)
        v_perp /= np.linalg.norm(v_perp)
        v_par = QE @ (QE.T @ v)
        v_par = v_par / max(np.linalg.norm(v_par), 1e-12)
        a = np.deg2rad(overlap_angle_deg)
        F[-1] = (np.cos(a) * v_par + np.sin(a) * v_perp).reshape(d, d)
        GF = flatten_basis(F)
        # 真值参数
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Cstar = sum(psi[j] * GF[:, j].reshape(d, d) for j in range(q))
        # 数据：A* 作用在 n_samples 个随机输入上
        U = rng.normal(size=(d, n_samples))
        A = sum(theta[i] * E[i] for i in range(p)) + Cstar
        Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
        G = np.column_stack([(Ei @ U).reshape(-1) for Ei in E] + [(Fj @ U).reshape(-1) for Fj in F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        theta_hat, psi_hat = sol[:p], sol[p:]
        C_hat = sum(psi_hat[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - Cstar, "fro"))
        # 预测：真修正项在 span(E) 上的投影（不可识别部分）
        proj = QE @ (QE.T @ GF @ np.array(psi))
        errs_pred.append(np.linalg.norm(proj))
        errs_theta.append(np.linalg.norm(theta_hat - theta))
    return np.mean(errs), np.mean(errs_pred), np.mean(errs_theta)


print("=" * 88)
print("[E1] 恢复误差 vs 先验类与修正类的主角度（p=3, q=3, sigma_noise=1e-3, 400 样本）")
print("   重叠角(deg)   实际主角度(deg)   ||Ĉ−C*||    预测 ||Π_E C*||    比值     ||θ̂−θ*||")
for deg in [0, 15, 30, 45, 60, 75, 90]:
    err, pred, et = run_case(3, 3, deg, 1e-3)
    print(f"   {deg:6d}        {deg:6d}          {err:.4e}    {pred:.4e}      {err/max(pred,1e-15):8.3f}   {et:.3e}")

print()
print("[E2] 噪声依赖（重叠角=30°，p=q=3）：噪声 -> 0 时误差是否收敛到投影预测值（结构偏差主导）")
for noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
    err, pred, et = run_case(3, 3, 30, noise, trials=20)
    print(f"   noise={noise:<8} ||Ĉ−C*||={err:.4e}   预测={pred:.4e}   比值={err/max(pred,1e-15):.3f}")

print()
print("[E3] 完全正交（90°）时是否可识别：噪声 -> 0 时 ||Ĉ−C*|| 是否 -> 0")
for noise in [1e-2, 1e-4, 1e-6, 1e-9]:
    err, pred, et = run_case(3, 3, 90, noise, trials=20)
    print(f"   noise={noise:<8} ||Ĉ−C*||={err:.4e}   预测={pred:.4e}   ||θ̂−θ*||={et:.3e}")

print()
print("[E4] 容量维度的影响（重叠角=45°, noise=1e-3）：q > p 时是否更不可识别？")
for p, q in [(3, 1), (3, 3), (3, 6), (6, 3)]:
    err, pred, et = run_case(p, q, 45, 1e-3, trials=20)
    print(f"   p={p}, q={q}: ||Ĉ−C*||={err:.4e}  预测={pred:.4e}  ||θ̂−θ*||={et:.3e}")

print()
print("判读：")
print(" * 若 E1/E2 的'实际/预测'比值 ~ O(1) 且 E3 在噪声→0 时收敛到 0，")
print("   则命题成立：**修正项的可识别性由 span(E) 与 span(F) 的主角度控制，")
print("   不可识别的部分恰是真修正项在 span(E) 上的投影**。")
print(" * 这给出一个可操作的架构设计律：让修正类与先验误差类尽量正交（主角度→90°），")
print("   而不是靠增大 beta / 加正则 —— 与 §4.5 的负面结论互补。")
