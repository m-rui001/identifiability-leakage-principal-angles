"""
B Agent / B-D6: 统一律的 **精确一阶形式** 与最坏情况（对抗性 misspecification）

背景（B-D5 的结论 + 一处自我纠正）：
  * 在 A 的构造里，err*sin(alpha)/||Delta|| 在 alpha=1..90 度、d=10 时 ≈ 0.16 —— A-CLAIM-16 复现。
  * 但 c1 * d = 1.75 跨 d=5..20 几乎不变 => c1 = 1.75/d，是**环境维数的稀释因子**，不是物理常数。
  * B-D5[M3] 我用 G 最小**右**奇异向量的先验分量构造 Delta，结果 err 反而小了 1000 倍 ——
    这说明"危险方向"不是 v_min（那是先验/修正可互换的系数方向，Delta 落进去会被 θ 完全吸收，
    对 ψ **零**伤害），而是 **左**奇异向量 u_min（响应空间里的近零方向）。这是我的错误，已修正。

本文件给出并验证精确机制：
    G = U_g diag(s) Vt  (瘦 SVD)，r = vec(Delta U + noise)
    psi_hat - psi  ≈  <r, u_min> / s_min * v_min
    ||Ĉ - C*||_F   ≈  |<r, u_min>| / s_min * ||sum_j v_min[j+p] F_j||_F          (*)
  可证伪：(*) 的预测 / 实测 ≈ 1（跨 alpha、跨随机/对抗 Delta）。
  由 (*)：放大因子 1/s_min(G) ~ 1/sin(alpha) 是**精确**的；而 c1 只是
  重叠因子 |<Delta U, u_min>|/(||Delta||·s_min 归一) —— 随机 Delta 时 ~1/d，对抗 Delta 时 ~O(1)。
  => 预测 ratio = err_adversarial / err_random ≈ d（量级），A 的平均情况律对结构化误差低估 d 倍。
"""
import numpy as np

rng = np.random.default_rng(31337)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(QE.shape[0], q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


def problem(d, p, q, alpha_deg, delta_norm, noise, n_samples=600, mode="random"):
    alpha = np.deg2rad(alpha_deg)
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    FB = build_F_at_angle(QE, p, q, alpha)
    F = [FB[:, j].reshape(d, d) for j in range(q)]
    theta = rng.normal(size=p); psi = rng.normal(size=q)
    U = rng.normal(size=(d, n_samples))
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])

    if mode == "adv":
        # 对抗：vec(Delta U) 沿 G 的最小左奇异向量 u_min
        Ug, sv, _ = np.linalg.svd(G, full_matrices=False)
        u = Ug[:, -1]
        W = u.reshape(d, n_samples)
        Dm = W @ U.T @ np.linalg.inv(U @ U.T)            # 解 Delta U = W 的最小范数 Delta
        Dm = Dm / np.linalg.norm(Dm, "fro") * delta_norm
    elif mode == "inspanE":
        # 对照：Delta 落在 span(E) 内（"漏掉的项其实是先验的一部分"）
        coef = rng.normal(size=p)
        Dm = sum(coef[i] * E[i] for i in range(p))
        Dm = Dm / np.linalg.norm(Dm, "fro") * delta_norm
    else:
        Dm = rng.normal(size=(d, d)); Dm = Dm / np.linalg.norm(Dm, "fro") * delta_norm

    A = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dm
    Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
    A0 = sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q))
    r = (Y - A0 @ U).reshape(-1)
    sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
    C_hat = sum(sol[p + j] * F[j] for j in range(q))
    Cm = sum(psi[j] * F[j] for j in range(q))
    actual = np.linalg.norm(C_hat - Cm, "fro")
    # 一阶预测 (*)
    Ug, sv, Vt = np.linalg.svd(G, full_matrices=False)
    v = Vt[-1]
    vp = v[p:]
    Fv = sum(vp[j] * F[j] for j in range(q))
    nf = np.linalg.norm(Fv, "fro")
    proj = abs(float(np.sum(r.reshape(-1) * Ug[:, -1])))
    pred = proj / sv[-1] * nf
    return actual, pred, sv[-1], proj / (delta_norm * np.sqrt(n_samples))


print("=" * 100)
print("[T1] 精确一阶公式 (*) 的预测/实测比（d=10,p=4,q=3,||D||=0.05,random Delta）")
print("  alpha    actual      pred(*)     pred/actual    s_min(G)     1/sin(a)")
for a in [2, 5, 10, 20, 45, 90]:
    acts, preds, sms = [], [], []
    for _ in range(12):
        ac, pr, sm, _ = problem(10, 4, 3, a, 0.05, 1e-4)
        acts.append(ac); preds.append(pr); sms.append(sm)
    ac, pr, sm = np.mean(acts), np.mean(preds), np.median(sms)
    print(f"  {a:5d}   {ac:.4e}   {pr:.4e}    {pr/ac:8.4f}     {sm:.3e}    {1/np.sin(np.deg2rad(a)):8.3f}"
          f"   1/(s_min*sin)归一={sm*1/np.sin(np.deg2rad(a)):.3e}")

print()
print("=" * 100)
print("[T2] 三种 Delta 方向：随机 / 落在 span(E) / 对抗(沿 u_min)。alpha=10, d=10")
print("  mode        actual      err*sin(a)/||D|| = c1    相对随机")
base = None
for mode in ["random", "inspanE", "adv"]:
    acts = [problem(10, 4, 3, 10.0, 0.05, 1e-4, mode=mode)[0] for _ in range(12)]
    ac = float(np.mean(acts))
    if mode == "random":
        base = ac
    c1 = ac * np.sin(np.deg2rad(10.0)) / 0.05
    print(f"  {mode:10s} {ac:.4e}     {c1:.4e}            {ac/base:8.2f}x")
print("  预测：inspanE << random（Delta 被 θ 吸收，ψ 几乎无伤）；adv ≈ d x random（各向同性稀释消失）")

print()
print("=" * 100)
print("[T3] 对抗情形的维数扫描：c1_adv 是否不再带 1/d？（预测 c1_adv*d / c1_rand*d ≈ 比值随 d 增大）")
print("   d    c1_rand   c1_adv   c1_adv/c1_rand   预测(≈d)")
for d in [5, 10, 20]:
    cr = np.mean([problem(d, 4, 3, 10.0, 0.05, 1e-4)[0] for _ in range(10)]) * np.sin(np.deg2rad(10)) / 0.05
    ca = np.mean([problem(d, 4, 3, 10.0, 0.05, 1e-4, mode="adv")[0] for _ in range(10)]) * np.sin(np.deg2rad(10)) / 0.05
    print(f"  {d:3d}   {cr:.4e}  {ca:.4e}     {ca/cr:8.2f}        {d}")

print()
print("=" * 100)
print("[T4] 交叉验证 1/d 机制：固定 d，改变 n_samples（随机 Delta）")
print("   重叠因子 |<Delta U,u_min>|/(||D||sqrt(n)) 应与 n 无关（若 ~1/d 纯属维数稀释则与 sqrt(n) 抵消）")
for ns in [150, 600, 2400]:
    ov, acts = [], []
    for _ in range(10):
        ac, pr, sm, o = problem(10, 4, 3, 10.0, 0.05, 1e-4, n_samples=ns)
        acts.append(ac); ov.append(o)
    print(f"  n_samples={ns:5d}: 重叠因子中位 = {np.median(ov):.4e}   err = {np.mean(acts):.4e}")

print()
print("[T5] 结论检查：若 c1_rand ≈ 1.75/d 且 c1_adv ≈ O(1)，则统一律应改写为")
print("   ||Ĉ-C*||  ~  ( |<Delta U, u_min>| + |<eps, u_min>| ) / sin(alpha_min)   （随机方向时额外 1/d 稀释）")
print("   设计含义：需要 sin(alpha_min) >> ||Delta||/d 才安全；对结构化 misspecification 门槛是 1 倍而非 1/d 倍。")
