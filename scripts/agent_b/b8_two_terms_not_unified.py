"""
B Agent / B-D8: "统一律"并不统一 —— 两项的概率性质不同（Δ 是确定性/对抗的，ε 是随机的）

§14.2 的写法   ‖Ĉ−C*‖ ≈ (c₁‖Δ‖ + c₂σ)/sin α_min   把两项放在同一个分子里、各配一个实测常数。
本轮结论：这是**类别错误**。由 L2/L3 的机制 σ_min(G)=√n·sin α/sqrt(1+cos²α/ρ_E²)：

  Δ 项（模型形式误差，**确定性**，方向可被物理选中）：
      最坏情况 = ‖Δ‖_F/sin α_min（常数恰 1，维数无关，B-D6 实测 0.999）
      平均情况 = ‖Δ‖_F/(d·sin α_min)（1.75/d 稀释，A 实测的 0.16 就是这个）
      ⇒ 必须用**最坏情况**：因为"漏掉的物理"不是随机的，它可以恰好落在 u_min 方向。

  ε 项（观测噪声，**独立高斯**，与 U 无关）：
      u_min 只依赖 U（设计），故条件于 U 时 ⟨vec(ε),u_min⟩ ~ N(0, σ²) 是**一维**高斯。
      ⇒ 以概率 ≥1-δ：  err_noise ≤ σ·z_{1-δ/2}/(√n·sin α_min)   —— 随 n 衰减，与 d 无关！
      对抗上界 σ√d/sinα（B-D7 的 L3 实测 √d 到 1%）只对"恶意噪声"合法，对 i.i.d. 噪声是空界。

⇒ 正确的、可发布的证书形式：
      ‖Ĉ−C*‖  ≤  [ ‖Δ_hidden‖ + σ·z_{1-δ/2}/√n ] / sin(α_min)          （ρ_E²≫1 时）
   两个"实测常数"c₁,c₂ 都被替换：c₁→1，c₂→z_{1-δ/2}/√n。
   含义：加数据能压掉噪声项，**压不掉模型形式误差项**（这条是 A §4.5 "调 β 无效"的定量加强）。

本文件核验：
  C1 ⟨vec(ε),u_min⟩/σ 的经验分布 = 标准正态（与 d,n、α 无关）
  C2 1-δ 分位与 z_{1-δ/2} 对照，给出证书里的实际数值
  C3 主导项随 (d,n,‖Δ‖/σ) 的切换：噪声项何时可忽略
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(9090)


def orth(Am):
    return np.linalg.qr(Am)[0]


def build_F_at_angle(QE, p, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(QE.shape[0], q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def u_min_of(d, p, q, alpha_deg, n_samples):
    alpha = np.deg2rad(alpha_deg)
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    FB = build_F_at_angle(QE, p, q, alpha)
    U = rng.normal(size=(d, n_samples))
    G = np.column_stack([((FB[:, j].reshape(d, d)) @ U).reshape(-1) for j in range(q)]
                        + [(Mi @ U).reshape(-1) for Mi in E])
    return np.linalg.svd(G, full_matrices=False)[0][:, -1]


print("=" * 100)
print("[C1] ⟨vec(ε),u_min⟩/σ 的经验分布（应 = 标准正态，与 d,n,α 无关）")
print("   d    n     alpha    均值     标准差   偏度    KS p值")
for d, ns, a in [(10, 600, 10), (20, 600, 10), (10, 2400, 10), (10, 600, 45), (6, 150, 2)]:
    u = u_min_of(d, 4, 3, a, ns)
    M = 4000
    z = (rng.normal(size=(M, d * ns)) @ u) / 1.0   # Var = ||u_min||^2 = 1 ⇒ 应 ~ N(0,1)
    ks = stats.ks_1samp(z, stats.norm.cdf)
    print(f"  {d:4d} {ns:6d} {a:6d}   {z.mean():+.4f}   {z.std():.4f}   {stats.skew(z):+.4f}   {ks.pvalue:.3f}")
print("  ⇒ 一维高斯确认：噪声项只贡献 σ·z_{1-δ/2}，除以 σ_min(G)≈√n·sinα/√(1+cos²α/ρ_E²) 后随 1/√n 消失。")

print()
print("=" * 100)
print("[C2] 证书里的噪声系数 c2(δ,n) = z_{1-δ/2}/√n  vs A 的实测 c2=0.063（平均情况）")
print("   n \\ δ     0.32(1σ)   0.05     0.001     （表内为 c2(δ,n)）")
for ns in [150, 600, 2400, 10000]:
    row = [stats.norm.ppf(1 - dl / 2) / np.sqrt(ns) for dl in [0.32, 0.05, 0.001]]
    print(f"  {ns:7d}    " + "   ".join(f"{v:.4f}" for v in row))
print("  A 的 c₂=0.063 落在 n=600 的 1σ 值 0.041 与 95% 值 0.080 之间 ⇒ **他的 c₂ 其实就是 z/√n，")
print("  不是常数**：把 n 从 600 改成 2400，c₂ 应当自动缩小 2 倍（可让 A 直接验证）。")

print()
print("=" * 100)
print("[C3] 两项谁主导：证书分子 = ‖Δ‖ + σ·c2(δ,n)，分母同为 sin α")
print("   ‖Δ‖   σ      n=600(95%) 的 c2σ    比值 ‖Δ‖/(c2σ)   结论")
for dn, sg in [(1e-3, 1e-2), (1e-2, 1e-2), (5e-2, 1e-2), (1e-2, 1e-1), (1e-2, 1e-3)]:
    c2 = stats.norm.ppf(0.975) / np.sqrt(600)
    print(f"  {dn:.0e}  {sg:.0e}      {c2*sg:.3e}          {dn/(c2*sg):8.2f}       "
          + ("噪声可忽略" if dn / (c2 * sg) > 10 else ("模型误差主导但噪声仍在" if dn / (c2 * sg) > 1 else "噪声主导")))
print("  ⇒ 实际 SciML 情形（‖Δ‖ 由'物理漏掉的项'决定，常 ≥1e-2）下模型形式误差项主导，")
print("    且它**不随数据量衰减**（常数 1），噪声项以 1/√n 衰减 ⇒ 大样本时证书由角度与 ‖Δ‖ 决定。")

print()
print("=" * 100)
print("[C4] 端到端：证书 R = [‖Δ‖ + σ z/√n]/sinα 对实测 ‖Ĉ−C*‖ 的覆盖（随机 Δ，非对抗）")
d, p, q, ns = 10, 4, 3, 600
rho_E = np.sqrt(d)
sa = np.sin(np.deg2rad(10))
fac = sa / np.sqrt(1 + np.cos(np.deg2rad(10)) ** 2 / rho_E ** 2)
for dn, sg in [(0.05, 1e-2), (0.01, 1e-2), (0.05, 1e-3)]:
    errs, cert = [], []
    for _ in range(40):
        alpha = np.deg2rad(10)
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E]); QE, _ = np.linalg.qr(GE)
        FB = build_F_at_angle(QE, p, q, alpha); F = [FB[:, j].reshape(d, d) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Dm = rng.normal(size=(d, d)); Dm = Dm / np.linalg.norm(Dm, 'fro') * dn
        U = rng.normal(size=(d, ns))
        A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dm
        Y = A @ U + rng.normal(scale=sg, size=(d, ns))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)[0]
        errs.append(np.linalg.norm(sum(sol[p + j] * F[j] for j in range(q))
                                   - sum(psi[j] * F[j] for j in range(q)), 'fro'))
        cert.append((dn + sg * stats.norm.ppf(0.975) / np.sqrt(ns)) / fac)
    errs = np.array(errs); cert = np.array(cert)
    # 注：证书分母用 sqrt(n)*fac = sigma_min(G)/sigma_min 归一（与 err 的系数空间尺度一致）
    print(f"  ‖Δ‖={dn:.2f}, σ={sg:.0e}: median err={np.median(errs):.3e}  证书原始值中位={np.median(cert):.3e}"
          f"  比值={np.median(cert)/np.median(errs):.2f}")
print("  ⇒ 证书/实测比值就是 **稀释因子**：随机 Δ 时 ~d（这里应≈10），对抗 Δ 时 →1。")
print("    论文若用随机 Δ 标定 c₁，则证书在最坏情况下被穿过 d 倍（B-D6 T2/T3 已实测）。")
