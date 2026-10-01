"""
B Agent / B-D7: 给 Prop 4.1 / 4.2 的**余项**做数值核验（把 §17.7 的草稿变成可信定理）

要核验的三个断言：
  L1（引理 A）  G^T G = n·Gop + O(√n)，Gop_ab = ⟨M_a,M_b⟩_F 为算子 Frobenius Gram。
                => ‖G^TG/n − Gop‖_F / ‖Gop‖_F 应按 n^{-1/2} 收敛。
  L2（引理 B / Prop 4.1）
                σ_min(G) = √n · sin α / sqrt(1 + cos²α/ρ_E²),  ρ_E = ‖E_i‖_F
                => 三个扫描：(a) 随 n（比值→1，误差 ~ n^{-1/2}）
                             (b) 随 ρ_E（把 E 整体缩放 s，预测公式应跟着动）
                             (c) 随 α（小 α 端最准，因为近零方向就是它）
  L3（Prop 4.2 的两项常数）
                最坏情况 ‖Ĉ−C*‖ ≈ (‖Δ‖_F + σ√d)/sin α
                Δ 项 T2 已测（c1_adv=0.999）。这里补 **噪声项**：把 ε 也对齐到 u_min，
                预测 err_noise·sin α/σ = √d。（若成立，A 的 c₂=0.063 就是平均情况，
                最坏情况是 σ√d —— 差 d·0.063·... 量级。）
"""
import numpy as np

rng = np.random.default_rng(4242)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(QE.shape[0], q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def make_G(d, p, q, alpha_deg, n_samples, scale_E=1.0):
    alpha = np.deg2rad(alpha_deg)
    E = [rng.normal(size=(d, d)) * scale_E / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    FB = build_F_at_angle(QE, p, q, alpha)
    F = [FB[:, j].reshape(d, d) for j in range(q)]
    U = rng.normal(size=(d, n_samples))
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
    Mats = E + F
    Gop = np.array([[np.sum(A * B) for B in Mats] for A in Mats])
    return G, Gop, FB, Mats, U


print("=" * 100)
print("[L1] 引理 A：G^T G / n -> Gop 的收敛率（d=10,p=4,q=3,alpha=10deg）")
print("    n_samples   rel_err = ||G^TG/n - Gop||_F/||Gop||_F     rel*sqrt(n)")
for ns in [50, 200, 600, 2400, 9600]:
    rels = []
    for _ in range(3):
        G, Gop = make_G(10, 4, 3, 10.0, ns)[:2]
        rels.append(np.linalg.norm(G.T @ G / ns - Gop, "fro") / np.linalg.norm(Gop, "fro"))
    rel = float(np.mean(rels))
    print(f"    {ns:8d}   {rel:.4e}                        {rel*np.sqrt(ns):.4f}")
print("  若最后一列近似常数 -> 引理 A 的 O(n^{-1/2}) 余项成立。")

print()
print("=" * 100)
print("[L2a] Prop 4.1 随 n：R = σ_min(G) / [√n · sinα / sqrt(1+cos²α/ρ_E²)]  应为 1+O(n^-1/2)")
d, p, q, a_deg = 10, 4, 3, 10.0
rho_E = np.sqrt(d)     # ‖E_i‖_F^2 = d^2/d = d（A 的 /sqrt(d) 归一化）
sa, ca = np.sin(np.deg2rad(a_deg)), np.cos(np.deg2rad(a_deg))
for ns in [50, 200, 600, 2400]:
    Rs = []
    for _ in range(8):
        G, Gop, *_ = make_G(d, p, q, a_deg, ns)
        sm = np.linalg.svd(G, compute_uv=False)[-1]
        Rs.append(sm / (np.sqrt(ns) * sa / np.sqrt(1 + ca**2 / rho_E**2)))
    print(f"    n={ns:5d}:  R = {np.mean(Rs):.4f}  (sd {np.std(Rs):.4f})   R-1 乘 √n = {(np.mean(Rs)-1)*np.sqrt(ns):+.3f}")

print()
print("[L2b] Prop 4.1 随字典缩放 s（ρ_E -> s·√d）：预测 R 仍为 1，但公式里 ρ_E 要跟着变")
print("     s      σ_min 实测      预测公式        比值")
for s in [0.1, 0.3, 1.0, 3.0, 10.0]:
    sms, preds = [], []
    for _ in range(8):
        G, Gop, *_ = make_G(d, p, q, a_deg, 600, scale_E=s)
        sm = np.linalg.svd(G, compute_uv=False)[-1]
        rho = s * rho_E
        sms.append(sm); preds.append(np.sqrt(600) * sa / np.sqrt(1 + ca**2 / rho**2))
    print(f"  {s:5.1f}   {np.mean(sms):.4f}      {np.mean(preds):.4f}     {np.mean(sms)/np.mean(preds):.4f}")
print("  含义：字典缩放会把 **系数空间** 的近零性藏起来（σ_min 变大），但算子空间的误差不变 ——")
print("        与 §17.2 的 M1 不变性实验一致；定理必须写成 σ_min 显式含 ρ_E 的形式。")

print()
print("[L2c] 随角度：n=600,d=10")
print("   alpha     σ_min 实测    预测      比值")
for a_deg_ in [2, 5, 10, 20, 45, 90]:
    sa_, ca_ = np.sin(np.deg2rad(a_deg_)), np.cos(np.deg2rad(a_deg_))
    sms = [np.linalg.svd(make_G(d, p, q, a_deg_, 600)[0], compute_uv=False)[-1] for _ in range(6)]
    pr = np.sqrt(600) * sa_ / np.sqrt(1 + ca_**2 / rho_E**2)
    print(f"   {a_deg_:5d}    {np.mean(sms):.4f}    {pr:.4f}   {np.mean(sms)/pr:.4f}")

print()
print("=" * 100)
print("[L3] Prop 4.2 噪声项：把 ε 对齐到 u_min，预测 err·sin α/σ = √d")
print("   d     σ        err(对齐ε)     err*sinα/σ     √d     预测/实测")
for dd in [5, 10, 20]:
    re_ = np.sqrt(dd)
    rh = np.sqrt(600) * np.sin(np.deg2rad(10)) / np.sqrt(1 + np.cos(np.deg2rad(10))**2/re_)
    out = []
    for _ in range(6):
        G, Gop, FB, Mats, U = make_G(dd, p, q, 10.0, 600)
        Ug = np.linalg.svd(G, full_matrices=False)[0]
        u = Ug[:, -1]
        sig = 1e-3
        W = u.reshape(dd, 600)
        eps = W / np.linalg.norm(W, "fro") * sig * np.sqrt(dd * 600)   # ||eps||_F = sig*sqrt(dn)
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        A0 = sum(theta[i] * Mats[i] for i in range(p)) + sum(psi[j] * Mats[p + j] for j in range(q))
        Y = A0 @ U + eps
        sol = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)[0]
        C_hat = sum(sol[p + j] * Mats[p + j] for j in range(q))
        Cm = sum(psi[j] * Mats[p + j] for j in range(q))
        out.append(np.linalg.norm(C_hat - Cm, "fro"))
    e = np.mean(out)
    print(f"  {dd:4d}   {sig:.0e}    {e:.4e}     {e*np.sin(np.deg2rad(10))/sig:.4f}   {re_:.3f}   {re_/ (e*np.sin(np.deg2rad(10))/sig):.3f}")
print("  判读：若实测 err*sinα/σ ≈ √d，则最坏情况噪声放大确实是 σ√d/sinα（A 的 c₂=0.063 只是")
print("        各向同性平均情况）；d=20 时两者差 ~45 倍，'常数'口径必须换成 √d。")
