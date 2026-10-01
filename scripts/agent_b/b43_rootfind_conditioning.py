"""
b43: is "obtainable to round-off by a monotone one-dimensional root find" (thm:exact, prop:multi)
true of the MATHEMATICS, or only of the tolerance I happened to use?

The chapter's sentence is about the reduction (a q x q monotone scalar problem replaces a (p+q)-dim
eigenproblem).  A referee will read the b42 table (worst relative deviation 1.6e-12) as evidence that
the *numerics* are free.  They are not, and the failure mode is predictable in advance:

  lam_min(G) ~ sin^2(alpha)/(1+cos^2(alpha) t)  ->  3.0e-10 at alpha = 0.01 deg,
  while a bisection whose stopping rule is ABSOLUTE (|upper-lo| < 1e-14) cannot resolve a root that
  small in relative terms: expected relative error ~ 1e-14/3e-10 ~ 3e-5.

So this script measures, per (alpha, kappa(Lam)) cell:
  (a) the ABSOLUTE-tolerance bisection, exactly as b42 runs it  -> the published claim's basis
  (b) a RELATIVE-tolerance bisection (tol = 1e-15 * upper)      -> the fix
  (c) eigvalsh(G) as arbiter.  Both are float64, so a disagreement localises to the root find.
Two questions, one of them against my own interest:
  Q1 does (a) lose relative accuracy as alpha -> 0?  (predicted yes, by 2-4 orders of magnitude)
  Q2 does (a) lose accuracy as kappa(Lam) -> inf?   (predicted: yes, and NOT fixable by tolerance --
      the resolvent (Lam-lam I)^{-1} is then computed from a nearly singular matrix, which is a
      conditioning limit of the REDUCTION, i.e. a real caveat for the chapter's sentence.)
"""
import numpy as np

rng = np.random.default_rng(91999)
D = 60
P = Q = 3


def lam_cond(kappa_target):
    """Gram of P unit-norm dictionary vectors with prescribed condition number."""
    E = rng.normal(size=(D, P))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    if kappa_target == 1.0:
        return E.T @ E
    # tilt: make the last direction weak -> kappa(Lam) ~ kappa_target
    E[:, -1] += np.sqrt(kappa_target - 1.0) * E[:, 0]
    E[:, -1] /= np.linalg.norm(E[:, -1])
    L = E.T @ E
    return L


def single_angle_blocks(L, alpha_deg):
    """Ch.4 construction: K = cos(a) * Lam^{1/2} Q, Q column-orthonormal -> equal cosine spectrum."""
    p, q = L.shape[0], Q
    ev, V = np.linalg.eigh(L)
    Lam_half = V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T
    R, _ = np.linalg.qr(rng.normal(size=(p, q)))
    return np.cos(np.deg2rad(alpha_deg)) * (Lam_half @ R)


def root_find(L, K, relative, iters=300, guard=1e-12):
    """eq:multi via bisection.  relative=False reproduces b42's ABSOLUTE stopping rule.
    guard: relative distance kept below the pole lam_min(L).  b42 used 1e-12, which is INSIDE the
    round-off zone of the pole (eps*||L||), where (L-lam I)^{-1} returns values of size ~1 rather
    than ~1/(lam_min(L)-lam).  Returns (lam, bracket_shrunk)."""
    hi = min(np.linalg.eigvalsh(L)[0], 1.0)
    Ip = np.eye(L.shape[0])

    def h(lam):
        try:
            return (1.0 - lam) - np.linalg.eigvalsh(K.T @ np.linalg.solve(L - lam * Ip, K))[-1]
        except np.linalg.LinAlgError:
            return np.inf

    shrunk = False
    for frac in (guard, 1e-8, 1e-5):
        upper_test = hi * (1.0 - frac)
        if np.isfinite(h(upper_test)):
            if frac > guard:
                shrunk = True
            break
        upper_test = None
    if upper_test is None:
        return np.nan, True
    lo, upper = 1e-16, upper_test
    if h(lo) <= 0.0:
        return 0.0, shrunk
    if h(upper) > 0.0:
        return upper, shrunk
    for _ in range(iters):
        mid = 0.5 * (lo + upper)
        if h(mid) > 0.0:
            lo = mid
        else:
            upper = mid
        gap = upper - lo
        if gap < (1e-15 * max(upper, 1e-300) if relative else 1e-14):
            break
    return 0.5 * (lo + upper), shrunk


print("=" * 104)
print("[Q1] alpha -> 0 at well-conditioned Lam.  relerr = |pred/eigvalsh(G) - 1|, 60 draws/cell.")
print("     The last column tests the mechanism: an ABSOLUTE error floor would make relerr*lam_min")
print("     constant (independent of alpha).")
print(f"   {'alpha(deg)':>11} {'lam_min (mean)':>15}  {'ABS tol (b42)':>14}  {'REL tol':>10}  {'max ABS':>10}  {'relerr*lam_min':>15}")
for a in [45.0, 5.0, 1.0, 0.1, 0.01, 0.001]:
    ea, er, tr = [], [], []
    for _ in range(60):
        L = lam_cond(1.0)
        K = single_angle_blocks(L, a)
        Gr = np.block([[L, K], [K.T, np.eye(Q)]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 0:
            continue
        tr.append(true)
        ea.append(abs(root_find(L, K, False)[0] / true - 1))
        er.append(abs(root_find(L, K, True)[0] / true - 1))
    print(f"   {a:11.3f} {np.mean(tr):15.3e}  {np.mean(ea):14.3e}  {np.mean(er):10.3e}  {max(ea):10.3e}"
          f"  {np.mean(np.array(ea) * np.array(tr)):15.3e}")

print()
print("[Q2] ill-conditioned dictionary (kappa(Lam) large) at fixed alpha=5deg.  'shrunk' = draws whose")
print("     bracket endpoint had to be pulled off min(lam_min(L),1) because the resolvent was singular.")
print(f"   {'kappa target':>13} {'kappa measured':>15}  {'ABS tol':>10}  {'REL tol':>10}  {'lam_min':>10}  {'shrunk/n':>9}")
for kt in [1.0, 1e3, 1e6, 1e9, 1e12]:
    ea, er, kk, tr, nsh, nn = [], [], [], [], 0, 0
    for _ in range(40):
        L = lam_cond(kt)
        K = single_angle_blocks(L, 5.0)
        Gr = np.block([[L, K], [K.T, np.eye(Q)]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 0:
            continue
        pa, sha = root_find(L, K, False)
        pb, shb = root_find(L, K, True)
        if np.isnan(pa) or np.isnan(pb):
            nsh += 1
            nn += 1
            continue
        nn += 1
        nsh += int(sha or shb)
        ev = np.linalg.eigvalsh(L)
        kk.append(ev[-1] / ev[0])
        tr.append(true)
        ea.append(abs(pa / true - 1))
        er.append(abs(pb / true - 1))
    if not ea:
        print(f"   {kt:13.1e}  (all {nn} draws: bracket unusable)")
        continue
    print(f"   {kt:13.1e} {np.mean(kk):15.3e}  {np.mean(ea):10.3e}  {np.mean(er):10.3e}  {np.mean(tr):10.3e}  "
          f"{nsh:5d}/{nn:<3d}")

print()
print("[Q2b] localising the Q2 failure: is it MY bracket, or the reduction?  One failing draw per cell.")
for kt in [1e3, 1e6, 1e12]:
    for attempt in range(40):
        L = lam_cond(kt)
        K = single_angle_blocks(L, 5.0)
        Gr = np.block([[L, K], [K.T, np.eye(Q)]])
        true = np.linalg.eigvalsh(Gr)[0]
        pred = root_find(L, K, True)[0]
        if not np.isfinite(true) or true <= 0 or abs(pred / true - 1) < 1.0:
            continue
        Ip = np.eye(L.shape[0])
        lp = np.linalg.eigvalsh(L)[0]
        hi = min(lp, 1.0)

        def hv(lam):
            try:
                return (1.0 - lam) - np.linalg.eigvalsh(K.T @ np.linalg.solve(L - lam * Ip, K))[-1]
            except np.linalg.LinAlgError:
                return np.inf

        print(f"   kappa~{kt:.0e}: true={true:.4e} pred={pred:.4e} lam_min(L)={lp:.4e} hi={hi:.4e}")
        print(f"      h(1e-16)={hv(1e-16):+.4e}  h(true)={hv(true):+.4e}  h(pred)={hv(pred):+.4e}"
              f"  h(hi*(1-1e-12))={hv(hi * (1 - 1e-12)):+.4e}")
        print(f"      eigvalsh(G) residual check: ||G v - true v||/||v|| =",
              f"{np.linalg.norm(Gr @ np.linalg.eigh(Gr)[1][:, 0] - true * np.linalg.eigh(Gr)[1][:, 0]):.3e}",
              f" cond(L)={np.linalg.cond(L):.2e}")
        break

print()
print("[Q2c] the same cells with the pole guarded (endpoint at lam_min(L)*(1-1e-6) instead of (1-1e-12)).")
print("     If Q2's failure was my bracket rather than the reduction, the errors must collapse to the")
print("     absolute floor of Q1, i.e. relerr ~ 1.7e-15/lam_min.  Last column: relerr*lam_min.")
print(f"   {'lam_min (mean)':>15}  {'relerr (guarded)':>17}  {'relerr*lam_min':>15}  {'1.7e-15/lam_min':>16}")
for kt in [1.0, 1e3, 1e6, 1e9, 1e12]:
    eg, tr = [], []
    for _ in range(40):
        L = lam_cond(kt)
        K = single_angle_blocks(L, 5.0)
        Gr = np.block([[L, K], [K.T, np.eye(Q)]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 0:
            continue
        pg = root_find(L, K, True, guard=1e-6)[0]
        if np.isnan(pg):
            continue
        eg.append(abs(pg / true - 1))
        tr.append(true)
    if not eg:
        print(f"   {'(no usable draw)':>15}")
        continue
    print(f"   {np.mean(tr):15.3e}  {np.mean(eg):17.3e}  {np.mean(np.array(eg) * np.array(tr)):15.3e}"
          f"  {1.7e-15 / np.mean(tr):16.3e}")
print()
print("[Q3] the same scan measured against the CLOSED FORM cf(t,alpha) instead of eigvalsh, to")
print("     separate 'my root find is loose' from 'the two are genuinely different objects'.")
print(f"   {'alpha(deg)':>11}  {'|cf/true-1|':>12}  {'|rootfind/true-1|':>18}  {'|rootfind/cf-1|':>16}")
for a in [45.0, 5.0, 1.0, 0.1, 0.01]:
    ec, em, ecx = [], [], []
    for _ in range(40):
        L = lam_cond(1.0)
        K = single_angle_blocks(L, a)
        Gr = np.block([[L, K], [K.T, np.eye(Q)]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 0:
            continue
        c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(L, K))[-1]
        t = np.linalg.norm(np.linalg.solve(L, K), 2) ** 2 / c2
        s = np.sin(np.deg2rad(a)) ** 2
        cf = ((1 + t) - np.sqrt(max((1 + t) ** 2 - 4 * t * s, 0.0))) / (2 * t)
        rf = root_find(L, K, True)[0]
        ec.append(abs(cf / true - 1))
        em.append(abs(rf / true - 1))
        ecx.append(abs(rf / cf - 1))
    print(f"   {a:11.3f}  {np.mean(ec):12.3e}  {np.mean(em):18.3e}  {np.mean(ecx):16.3e}")

print()
print("判读（实测，非预登记的期望）：")
print("  Q1  两个停止准则的**相对**误差都随 alpha 变小而上升（ABS 6.4e-15 -> 1.4e-5，REL 8.1e-16 -> 1.6e-6），")
print("     我预登记的'REL 列保持 1e-14'不成立。真正不变的量是最后一列：relerr*lam_min 恒在")
print("     [1.55,1.84]e-15 跨六个数量级，即误差是**绝对**地板 ~1.7e-15*||G||，相对精度 ~1/sin^2 alpha。")
print("     所以正文那句'到舍入误差可得'必须写成**绝对**舍入误差；b42 的 1.6e-12 只是中等角度下的")
print("     相对读数，不能外推到 alpha->0。")
print("  Q2  REL 列确实随 kappa(Lam) 上升（3.3 -> 82），但 Q2b 证明这不是约化本身的极限： pred 恰等于")
print("     lam_min(L)（=区间端点），而 h 在 true 处 +3.4e-13、在端点处 +8.5e-4，eigvalsh 残差 1e-16 ——")
print("     是我把端点取进了预解式的奇点邻域（(Lam-lam)^-1 近奇异），**我的括号 bug**，数学无恙。")
print("  Q2c 端点内缩到 lam_min(L)*(1-1e-6) 之后地板回来了：lam_min 3.4e-3/3.7e-6/3.7e-9 三档")
print("     relerr*lam_min = 1.9e-16/2.0e-16/1.7e-16。但 lam_min 3.7e-12/3.6e-15 两档 relerr = 16/76，")
print("     任何 guard 都救不回来 —— 那里 λ 已低于 eps*||Lam||，'绝对地板'不再是可用的数。")
print("     预登记的'换停止准则救不回来'对中段是错的（guard 就是解），只对最深两档成立。")
print("  Q3  三列在同一数量级上相等（|cf/true-1| ~ |rootfind/true-1| ~ |rootfind/cf-1|，1.5e-15..3e-8），")
print("     所以 b42 里小角度处的 cf 与根之差**就是数值地板本身**，不是 cf 与根两个对象的真实间隔；")
print("     与'我的求根松'也区分不开——三者松紧一致。这不动 (★)：cf 在证书里只走上界侧。")
