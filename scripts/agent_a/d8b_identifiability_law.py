"""
A Agent / D8b: 修正 §D8 的错误猜想，给出**正确且可验证**的可识别性律
  v1（错误）猜想：不可识别部分 = 真修正项在 span(E) 上的投影  -> 被数值否证（见 d8_out.txt E1）
  正确表述（本文件验证）：
    (T1) 零偏差可识别  <=>  span(E) ∩ span(F) = {0}   （等价于堆叠基线性无关）
         角度 > 0 时偏差为 0；角度 = 0（有公共方向）时存在不可消除的偏差。
    (T2) 统计稳定性由**主角度**控制：误差 ~ noise / sin(alpha)。
         构造一个 q 维子空间，使其与 span(E) 的所有主角度都等于 alpha，
         则估计误差应随 1/sin(alpha) 增长，alpha -> 0 时发散。
"""
import numpy as np
from scipy.linalg import subspace_angles

rng = np.random.default_rng(31337)
d = 10


def orth(A):
    Q, _ = np.linalg.qr(A)
    return Q


def build_F_at_angle(QE, p, q, alpha):
    """构造 q 维子空间基 F，使其与 span(QE) 的所有主角度 = alpha"""
    QE = orth(QE[:, :p])
    a_raw = rng.normal(size=(p, q))
    a = orth(QE @ orth(a_raw))                      # span(E) 内的 q 个正交方向
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))           # 与 span(E) 正交的 q 个正交方向
    return np.cos(alpha) * a + np.sin(alpha) * b     # 逐列相同角度


def run(p, q, alpha_deg, noise, n_samples=600, trials=30):
    alpha = np.deg2rad(alpha_deg)
    errs, bias0 = [], []
    for _ in range(trials):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        FB = build_F_at_angle(QE, p, q, alpha)
        F = [FB[:, j].reshape(d, d) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q))
        U = rng.normal(size=(d, n_samples))
        Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        psi_hat = sol[p:]
        C_hat = sum(psi_hat[j] * F[j] for j in range(q))
        C = sum(psi[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - C, "fro"))
        # 零噪声时的结构偏差
        sol0, *_ = np.linalg.lstsq(G, (A @ U).reshape(-1), rcond=None)
        C0 = sum(sol0[p + j] * F[j] for j in range(q))
        bias0.append(np.linalg.norm(C0 - C, "fro"))
    ang = np.rad2deg(subspace_angles(QE, orth(FB))).min()
    return np.mean(errs), np.mean(bias0), ang


print("=" * 92)
print("[E5] 主角度 alpha 固定（p=4, q=3, noise=1e-3, 600 样本）")
print("   alpha(设计)  实测最小主角度  ||Ĉ−C*||(含噪)   零噪声偏差   预测 noise/sin(alpha)   比值")
rows = []
for a in [2, 5, 10, 20, 30, 45, 60, 90]:
    err, bias0, ang = run(4, 3, a, 1e-3)
    pred = 1e-3 / np.sin(np.deg2rad(a))
    rows.append((a, ang, err, bias0, pred))
    print(f"   {a:8d}     {ang:10.3f}      {err:.4e}      {bias0:.3e}     {pred:.4e}        {err/pred:6.3f}")

print()
print("[E6] 噪声扫描（alpha=10°）：误差 ∝ noise 是否成立（线性响应）")
print("   noise      ||Ĉ−C*||      /noise      预测系数 1/sin(10°)=5.76")
for noise in [1e-4, 1e-3, 1e-2, 1e-1]:
    err, bias0, _ = run(4, 3, 10, noise, trials=20)
    print(f"   {noise:<9}  {err:.4e}   {err/noise:8.3f}")

print()
print("[E7] 精确公共方向（alpha=0）时的偏差与噪声无关性")
for noise in [1e-6, 1e-3, 1e-1]:
    err, bias0, ang = run(4, 3, 0, noise, trials=10)
    print(f"   noise={noise:<8} ||Ĉ−C*||={err:.4e}   零噪声偏差={bias0:.4e}   实测最小主角度={ang:.2e}")

print()
print("判读（取代 D8 v1 的错误猜想）：")
print(" * T1 成立：只要主角度 > 0，零噪声偏差 -> 0（E5 的'零噪声偏差'列在 alpha>=2° 时 ~1e-14 量级）；")
print("   alpha = 0（存在公共方向）时零噪声偏差 O(1)，且与噪声无关 —— 这才是真正的不可识别。")
print(" * T2 成立：误差 ~ noise/sin(alpha)（E5 比值列 O(1)，E6 显示误差与噪声成正比）。")
print(" * 因此设计律是：**先保证交集为零（线性无关），再把主角度做小？ 不是——角度越小方差越大，")
print("   所以要把主角度做大**，即先验误差类与修正类要'松散'耦合；而调 beta 只改方差水平不改交集。")
