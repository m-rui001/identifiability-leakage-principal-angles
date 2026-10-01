"""
A Agent / D10 (S3 x misspecification): 把"噪声"换成"模型形式误差"后，1/sin(alpha) 律是否仍成立？
  设定：真算子  A = P(θ*) + C(ψ*) + Δ_hidden
        Δ_hidden 落在两个类之外（模拟'物理模型漏掉的东西'），拟合只用 P + C
  预测（本文件验证）：修正项恢复误差 ≈ c' * ||Δ_hidden||_F / sin(alpha_min)
  意义：这把 S3（可识别性）与郭玲组 2603.24948 / 2606.03469 的 misspecification 主战场对接——
        修正网络的保真度由主角度与模型误差量级共同决定，且**不能靠调 beta**。
"""
import numpy as np

rng = np.random.default_rng(60601)
d = 10


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def run(p, q, alpha_deg, delta_norm, noise, n_samples=600, trials=30):
    alpha = np.deg2rad(alpha_deg)
    errs = []
    for _ in range(trials):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        FB = build_F_at_angle(QE, p, q, alpha)
        F = [FB[:, j].reshape(d, d) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * delta_norm
        A = (sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh)
        U = rng.normal(size=(d, n_samples))
        Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        C_hat = sum(sol[p + j] * F[j] for j in range(q))
        C = sum(psi[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - C, "fro"))
    return np.mean(errs)


print("=" * 92)
print("[G1] 模型形式误差 Δ_hidden 作为'扰动源'：修正项恢复误差 vs 主角度（p=4,q=3,σ_noise=1e-4）")
print("   alpha    ||Δ||=0.01              ||Δ||=0.05              ||Δ||=0.1       预测 c*||Δ||/sin(a)")
for a in [2, 5, 10, 20, 30, 45, 60, 90]:
    row = []
    for dn in [0.01, 0.05, 0.1]:
        row.append(run(4, 3, a, dn, 1e-4, trials=15))
    pred = row[-1] / (0.1 / np.sin(np.deg2rad(a)))   # 用 ||Δ||=0.1 的实测反推常数 c*
    print(f"   {a:5d}   {row[0]:.4e}   {row[1]:.4e}   {row[2]:.4e}      {pred:.4f}")

print()
print("[G2] 定量检验：err*sin(alpha)/||Δ|| 是否为常数（跨 alpha 与 ||Δ|| 同时检验）")
print("   alpha   ||Δ||=0.01    ||Δ||=0.05    ||Δ||=0.1   （理想：三列相等且不随 alpha 变）")
for a in [2, 5, 10, 20, 45, 90]:
    vals = [run(4, 3, a, dn, 1e-4, trials=15) * np.sin(np.deg2rad(a)) / dn for dn in [0.01, 0.05, 0.1]]
    print(f"   {a:5d}    {vals[0]:.4e}    {vals[1]:.4e}    {vals[2]:.4e}")

print()
print("[G3] 与纯噪声情形对照（Δ_hidden=0, σ_noise=1e-2）：err*sin(alpha)/σ 是否也是常数")
for a in [2, 5, 10, 20, 45, 90]:
    v = run(4, 3, a, 0.0, 1e-2, trials=15) * np.sin(np.deg2rad(a)) / 1e-2
    print(f"   alpha={a:3d}: err*sin(a)/σ = {v:.4e}")

print()
print("判读：若 G2 三列近似相等且跨 alpha 稳定 -> 命题成立：")
print("  修正项恢复误差 ≈ c * (噪声 + 模型形式误差) / sin(alpha_min)")
print("即：**主角度把两类误差（随机估计误差与模型形式误差）按同一 1/sin 因子放大**。")
