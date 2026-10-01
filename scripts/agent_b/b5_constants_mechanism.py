"""
B Agent / B-D5: 在 **A 自己的构造** 里核查 A-CLAIM-16 的两个常数，并给出机制

A/d10_misspec_angle.py 的 build_F_at_angle 我原样保留（正交标架、等角），只加三件事：

  M1 重参数化不变性（免费正确性检查）：把先验基 E 整体乘 s。
     模型空间 span{E,U} 不变，OLS 投影不变 => ||Ĉ-C||_F 必须与 s 无关。
     若不变性成立而 c1 随 **相对量级** ||Δ||/||A_model|| 变，则 c1 不是"物理常数"，
     它携带了 Δ 相对于模型的能量比 —— 单位都不干净。
  M2 维数稀释（B-D4 里发现的 c1 ~ 0.3/d）：在 A 的构造里重做 d 扫描，确认不是我的构造伪影。
  M3 机制 = **对齐因子**：c1 = ||P_{vmin} vec(ΔU)|| / ||vec(ΔU)||，其中 vmin 是设计矩阵
     最小奇异方向。随机 Δ 时它 ~ 1/sqrt(d^2)；把 Δ **做成沿 vmin 方向**（对抗/结构化 misspecification）
     时它 ~ O(1)，即误差比 A 的 c1 大 d^2/... 倍。若实测比值 ≈ d，则 A 的律是"平均情况"律，
     对'物理漏掉的项恰好落在先验与修正可互换方向'这一最危险情形低估 d 倍。

  可证伪预测：
    (a) c1(d) * d = 常数
    (b) err(对齐 Δ) / err(随机 Δ) ≈ d  （= √(d²)/1）
    (c) err 与 s 无关（重参数化不变）
"""
import numpy as np

rng = np.random.default_rng(60601)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha):
    """A 的原文构造：F 的 q 个方向与 span(E) 成同一主角度 alpha，且彼此正交"""
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(QE.shape[0], q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def make_problem(d, p, q, alpha_deg, delta_norm, noise, scale_E=1.0, n_samples=600,
                 aligned=False, seed_off=0):
    alpha = np.deg2rad(alpha_deg)
    E = [rng.normal(size=(d, d)) * scale_E / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    FB = build_F_at_angle(QE, p, q, alpha)
    F = [FB[:, j].reshape(d, d) for j in range(q)]
    theta = rng.normal(size=p); psi = rng.normal(size=q)
    U = rng.normal(size=(d, n_samples))
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
    sv = np.linalg.svd(G, compute_uv=False)
    if aligned:
        # 让 Δ 沿 "能把 ψ 换进 θ 的近零方向"：取 G 最小奇异方向（右奇异向量）v，
        # v 分解为 (dtheta, dpsi) 两部分；构造 Δ = -(Σ dtheta_i E_i) 使模型里
        # "先验与修正可互换" 的分量恰好被 Δ 激发。
        Vt = np.linalg.svd(G, full_matrices=False)[2]
        v = Vt[-1]
        dth = v[:p]
        nrm = np.linalg.norm(sum(dth[i] * E[i] for i in range(p)), "fro")
        Dh = -sum(dth[i] * E[i] for i in range(p)) / nrm * delta_norm
    else:
        Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * delta_norm
    A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh
    Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
    sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
    C_hat = sum(sol[p + j] * F[j] for j in range(q))
    Cm = sum(psi[j] * F[j] for j in range(q))
    return np.linalg.norm(C_hat - Cm, "fro"), sv[-1] / sv[0]


def avg(d, p, q, a, dn, noise, trials=15, **kw):
    errs = [make_problem(d, p, q, a, dn, noise, **kw)[0] for _ in range(trials)]
    return float(np.mean(errs))


amp = np.sin(np.deg2rad(10.0))
print("=" * 96)
print("[M1] 重参数化不变性：先验基整体缩放 s（模型空间不变 -> err 必须不变）")
print("      s        err")
for s in [0.01, 0.1, 1.0, 10.0]:
    e = avg(10, 4, 3, 10.0, 0.05, 1e-4, scale_E=s)
    print(f"   {s:8.2f}   {e:.4e}")
print("   判读：err 应几乎不变（OLS 对列缩放的不变性）。若成立，则后续任何 c1 变化都来自")
print("   **Δ 相对模型的能量比 / 维数 / 方向**，而不是模型类别本身。")

print()
print("=" * 96)
print("[M2] A 的构造里的维数扫描（p=4,q=3,alpha=10deg,||D||=0.05,随机方向）")
print("     d       err         c1=err*sin a/||D||     c1*d      kappa(G)")
for d in [5, 7, 10, 14, 20]:
    errs, kn = [], []
    for _ in range(15):
        e, k = make_problem(d, 4, 3, 10.0, 0.05, 1e-4)
        errs.append(e); kn.append(k)
    e = float(np.mean(errs)); c1 = e * amp / 0.05
    print(f"  {d:4d}   {e:.4e}      {c1:.4e}        {c1*d:.4f}    {np.median(kn):.3e}")
print("   预测 (a)：c1*d 近似常数 -> A 的 c1 是 1/d 稀释因子，d=10 时 '0.16' 无普适性。")

print()
print("=" * 96)
print("[M3] 对齐 Δ（misspecification 恰好落在先验/修正可互换方向）vs 随机 Δ")
print("     d    err_random    err_aligned    ratio      预测 ratio ~ d")
for d in [5, 10, 20]:
    er = avg(d, 4, 3, 10.0, 0.05, 1e-4)
    ea = avg(d, 4, 3, 10.0, 0.05, 1e-4, aligned=True)
    print(f"  {d:4d}   {er:.4e}     {ea:.4e}      {ea/er:8.2f}     {d}")
print("   判读：若 ratio ~ d，则 A 的统一律只对 **各向同性（随机）模型形式误差** 成立；")
print("        结构化误差（例如漏掉的项与某个先验算子对称性相同）会再放大 d 倍。")

print()
print("=" * 96)
print("[M4] 1/sin alpha 律在 A 的构造里的 **有效区间**（d=10,p=4,q=3,||D||=0.05）")
print("   alpha    err        err*sin(a)/||D||    1/sin(a) 归一预测(用10deg锚定)")
base = None
for a in [1, 2, 5, 10, 20, 40, 60, 90]:
    e = avg(10, 4, 3, a, 0.05, 1e-4)
    if a == 10:
        base = e
    sa = np.sin(np.deg2rad(a))
    anchor = base * np.sin(np.deg2rad(10.0)) / sa if base else float("nan")
    print(f"  {a:5d}   {e:.4e}     {e*sa/0.05:.4e}         {anchor:.4e}   比值={e/anchor if anchor else float('nan'):.3f}")
print("   判读：若在 **大角度** 端 err*sin(a) 不再守恒（出现误差地板），则 1/sin 律只是小角度")
print("        渐近式；论文里必须写明适用条件 alpha <~ 某个由 d、n 决定的阈值。")

print()
print("=" * 96)
print("[M5] 对齐情形下的角度律：c1_aligned 是否 ~ O(1)（即不再需要 1/d）")
for a in [2, 10, 30]:
    e = avg(10, 4, 3, a, 0.05, 1e-4, aligned=True)
    print(f"  alpha={a:3d}: err*sin(a)/||D|| = {e*np.sin(np.deg2rad(a))/0.05:.4e}")
print("   若这些值 ~O(1) 且跨 alpha 稳定，则正确的写法是")
print("     ||Ĉ-C*|| ≈ ||P_vmin (Δ U)|| / sin(alpha_min)，  随机 Δ 时 ||P_vmin|| ~ ||Δ||·√(n)/d")
print("   —— 角度放大是 **精确机制**，常数只是投影重叠，随误差的方向性变化 d 倍。")
