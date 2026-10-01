"""
b49 -- B-CLAIM-55(a): predict the family's gain over cf1 from the TWO-JET of lam(t) along the
steepest-descent ray, so that "is cf1 good enough?" becomes decidable without optimising.

b48 settled the qualitative question: cf1 is not the minimum of the family (20), it is a saddle,
because along v(t) = cos t v1 + sin t u (u perp v1) the first variation of C is 2 v1^T B2 u != 0
while that of B vanishes. Its E section measured the dip numerically (84/84 draws, removing
0.07-0.50 of cf1's over-estimate). This script turns that into a FORMULA, derived here before any
number is produced, so that the match is a test and not a fit.

THE DERIVATION.  F(lam;B,C) = C lam^2 - (B+C) lam + B - B^2 = 0, and along the ray
    B(t) = B0 + b2 t^2 + O(t^4)      (b2 = B(u) - B(v1); NO linear term: A0 v1 = c^2 v1, u perp v1)
    C(t) = C0 + c1 t  + c2 t^2 + O(t^3)   (c1 = 2|P_perp B2 v1| by choice of u, c2 = C(u) - C(v1))
with lam(t) = lam0 + L1 t + L2 t^2.  Total second derivative of F = 0 at t = 0, using
    F_lam = 2C lam - B - C,   F_lam lam = 2C,   F_B = 1 - lam - 2B,   F_C = lam^2 - lam,
    F_lam B = -1,   F_lam C = 2 lam - 1,   F_C C = F_B C = 0   (F is LINEAR in C),
    B'(0) = 0,  B''(0) = 2 b2,  C'(0) = c1,  C''(0) = 2 c2,
gives
    L1 = -F_C c1 / F_lam,
    L2 = -[ F_lam lam L1^2 + 2 F_lam C L1 c1 + 2 F_B b2 + 2 F_C c2 ] / (2 F_lam).
At the SMALLER root F_lam < 0, and F_C = lam(lam-1) < 0 for lam in (0,1), so L1 = c1 lam(1-lam)/F_lam
< 0: descent.  If additionally L2 > 0 the ray has an interior minimum at
    t_pred = -L1/(2 L2),        dip_pred = -L1^2/(4 L2)          (both O(c1^2), i.e. QUADRATIC in
the off-E0 gradient even though the descent itself is LINEAR in t).

PRE-REGISTERED, before running:
  A  dip_measured/dip_pred -> 1 with the residual O(c1^3): ratio in [0.9,1.1] wherever |c1| is small,
     and drifting below 1 as |c1| grows (the ray is only steepest to FIRST order, so a large c1 makes
     the u-choice suboptimal and the true dip DEEPER than predicted => ratio > 1 is the falsifier for
     the bookkeeping, ratio < 1 with growing |c1| is the expected suboptimality of u).
     FALSIFIER of the derivation: ratio bounded away from 1 at small |c1| (say 1.3 or 0.7) => my
     implicit second derivative is wrong and there is no formula here.
  B  t_measured/t_pred -> 1, same tolerance; this is the sharper test because t* = -L1/2L2 has no
     square of a small quantity in it.
  C  L2 > 0 in every draw.  If L2 <= 0 somewhere the ray has no interior minimum and its dip is a
     boundary effect of my t-grid -- that draw is then uninformative, not a contradiction.
  D  prop:chord must hold on the refined ray (viol = 0), and the dip must never push lam(v) below
     lam_min(G).  A resolved violation would reopen the chapter.
  E  cross-check against a SEARCH, not a sample: on the SAME draws a polar net around v1 over the
     whole sphere (64 directions x 64 radii, always containing the steepest ray) gives min_v lam(v).
     If the net regularly beats the ray, u is not the right direction and the formula underestimates
     the available gain -- which is exactly what I need to know to stop.  (b48's net used a different
     seed, so it cannot be compared draw by draw; this net is re-run here for that reason.)
  F  the formula must never predict MORE gain than exists: dip_pred >= -(cf1 - lam_min(G)) in every
     draw, since no family member is below lam_min(G) by prop:chord.  A violation is a bug in the
     two-jet algebra or an L2 that has gone unphysically small.
"""
import numpy as np

rng = np.random.default_rng(50512)
NET = np.random.default_rng(9001)          # SEPARATE stream: the (E) net must not move the draws
N = 40
EPS = np.finfo(float).eps

SPECTRA = {
    "spread 5-75":     [np.cos(np.deg2rad(a)) for a in (5, 40, 75)],
    "spread 2-45-88":  [np.cos(np.deg2rad(a)) for a in (2, 45, 88)],
    "spread 30-88-89": [np.cos(np.deg2rad(a)) for a in (30, 88, 89)],
    "equal 20d":       [np.cos(np.deg2rad(20))] * 3,
}


def lam_spd(p):
    E = rng.normal(size=(100, p))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E.T @ E


def blocks(Lam, cosines):
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))
    M = Q1 @ np.diag(cosines) @ Q2.T
    ev, V = np.linalg.eigh(Lam)
    return (V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T) @ M


def cf_of(t, c2):
    s = 1.0 - c2
    disc = max((1.0 + t) ** 2 - 4.0 * s * t, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t)


def crossings(Bs, Cs):
    disc = (Bs + Cs) ** 2 - 4.0 * Cs * Bs * (1.0 - Bs)
    bad = disc < 0.0
    disc = np.where(bad, 0.0, disc)
    out = np.where(Cs > 0.0, ((Bs + Cs) - np.sqrt(disc)) / (2.0 * np.where(Cs > 0, Cs, 1.0)), 1.0)
    return np.where(bad, 1.0, out)


def lam_of_v(v, A0, B2):
    v = v / np.linalg.norm(v)                      # every direction here must be ON the sphere
    return float(crossings(np.array([float(v @ A0 @ v)]), np.array([float(v @ B2 @ v)]))[0])


def perp_net(v1, nu):
    """(nu,q) unit rows spanning v1^perp -- b48's lesson: build them from an SVD of I-v1v1^T and
    never normalise a machine-zero vector.  q=2 has only two directions, both already unit."""
    q = v1.size
    U, s, Vt = np.linalg.svd(np.eye(q) - np.outer(v1, v1))
    Bv = Vt[: q - 1].T
    if q == 2:
        return np.array([Bv[:, 0], -Bv[:, 0]])     # S^0 = {+b,-b}: both sides, positive t only
    if q == 3:
        th = np.linspace(0.0, 2 * np.pi, nu, endpoint=False)
        return np.outer(np.cos(th), Bv[:, 0]) + np.outer(np.sin(th), Bv[:, 1])
    R = NET.normal(size=(nu, q - 1))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R @ Bv.T


def net_dip(v1, A0, B2, cf1, nu=64, nt=64):
    """(E) the same-draw competitor detector: a polar net around v1 over the whole sphere, i.e. a
    search that is NOT restricted to the steepest ray.  Returns (min over net of lam - cf1, #net
    points below lam_min of the 2x2 chord model is checked by the caller through 'true')."""
    Us = perp_net(v1, nu)
    tn = np.logspace(-6.0, np.log10(0.5 * np.pi), nt)
    Vs = (np.cos(tn)[:, None, None] * v1[None, None, :]
          + np.sin(tn)[:, None, None] * Us[None, :, :]).reshape(-1, v1.size)
    Vs /= np.linalg.norm(Vs, axis=1, keepdims=True)
    Bv = np.einsum('ij,jk,ik->i', Vs, A0, Vs)
    Cv = np.einsum('ij,jk,ik->i', Vs, B2, Vs)
    return float(np.min(crossings(Bv, Cv)) - cf1), int(np.max(np.abs(np.linalg.norm(Vs, axis=1) - 1.0)))


def setup(Lam, K):
    """verbatim from b48 (no __main__ guard there, so importing is not an option)."""
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = np.linalg.eigvalsh(Gr)[0]
    A0 = K.T @ np.linalg.solve(Lam, K)
    B2 = K.T @ np.linalg.solve(Lam, np.linalg.solve(Lam, K))
    w0, V0 = np.linalg.eigh(A0)
    c2 = w0[-1]
    if c2 > 1.0 + 1e-12 or c2 <= 0.0:
        return None
    tol = 1e-9 * max(abs(c2), 1.0)
    E0 = V0[:, w0 >= w0[-1] - tol]
    mu, W = np.linalg.eigh(E0.T @ B2 @ E0)
    t1 = float(mu[-1]) / c2
    cf1 = cf_of(t1, c2)
    v1 = E0 @ W[:, -1]
    v1 = v1 / np.linalg.norm(v1)
    lbar = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    if cf1 >= lbar:
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, t1=t1, cf1=cf1, v1=v1, m0=E0.shape[1],
                floor=100.0 * EPS * np.linalg.norm(Gr, 2))


def two_jet(s):
    """L1, L2, t_pred, dip_pred from the derivation above, plus the coefficients they use."""
    v1, A0, B2, lam, c2 = s["v1"], s["A0"], s["B2"], s["cf1"], s["c2"]
    g = B2 @ v1
    gp = g - v1 * (v1 @ g)
    gp = gp - v1 * (v1 @ gp)
    nrm = float(np.linalg.norm(gp))
    if nrm < 1e-12 * float(np.linalg.norm(g)):                     # b48's lesson: do NOT normalise noise
        return None
    u = gp / nrm
    B0, C0 = c2, float(v1 @ B2 @ v1)
    b2 = float(u @ A0 @ u) - B0                                    # coefficient of t^2 in B
    c1, c2t = 2.0 * nrm, float(u @ B2 @ u) - C0                    # t and t^2 coefficients in C
    Fl = 2.0 * C0 * lam - B0 - C0                                  # < 0 at the smaller root
    Fll, FB, FC, FlC = 2.0 * C0, 1.0 - lam - 2.0 * B0, lam * lam - lam, 2.0 * lam - 1.0
    L1 = -FC * c1 / Fl
    L2 = -(Fll * L1 * L1 + 2.0 * FlC * L1 * c1 + 2.0 * FB * b2 + 2.0 * FC * c2t) / (2.0 * Fl)
    tp = -L1 / (2.0 * L2) if L2 > 0 else np.nan
    dip = -L1 * L1 / (4.0 * L2) if L2 > 0 else np.nan
    return dict(u=u, nrm=nrm, c1=c1, L1=L1, L2=L2, tp=tp, dip=dip, m0=s["m0"],
                origin_err=abs(lam_of_v(v1, A0, B2) - lam))


def run_cell(tag, gen, n=N):
    rows = []
    for _ in range(n):
        s = setup(*gen())
        if s is None:
            continue
        tj = two_jet(s)
        if tj is None:
            rows.append(dict(stat=True, m0=s["m0"]))
            continue
        A0, B2, v1, u, true, floor = s["A0"], s["B2"], s["v1"], tj["u"], s["true"], s["floor"]
        cf1 = s["cf1"]
        ts = np.unique(np.concatenate((                           # dense around t_pred, wide overall
            tj["tp"] * np.logspace(-2.0, 2.0, 600) if tj["tp"] == tj["tp"] and tj["tp"] > 0 else [],
            np.logspace(-7.0, np.log10(0.5 * np.pi), 1200))))
        lams = np.array([lam_of_v(np.cos(t) * v1 + np.sin(t) * u, A0, B2) for t in ts])
        ue = float(np.max(np.abs(np.linalg.norm(
            np.cos(ts)[:, None] * v1[None, :] + np.sin(ts)[:, None] * u[None, :], axis=1) - 1.0)))
        j = int(np.argmin(lams))
        nd, nue = net_dip(v1, A0, B2, cf1)
        rows.append(dict(stat=False, m0=s["m0"], nrm=tj["nrm"], c1=tj["c1"], L1=tj["L1"], L2=tj["L2"],
                         tp=tj["tp"], dip=tj["dip"], t_meas=float(ts[j]), dip_meas=float(lams[j] - cf1),
                         viol=int(np.sum((lams < true) & (true - lams > floor))), ue=max(ue, nue),
                         origin_err=tj["origin_err"], gap=cf1 - true, net=nd, floor=floor,
                         over=int(tj["dip"] == tj["dip"] and tj["dip"] < -(cf1 - true))))
    st = [r for r in rows if r["stat"]]
    act = [r for r in rows if not r["stat"]]
    print(f"   {tag:>26}: n={len(rows):3d} dimE0={sorted({r['m0'] for r in rows})}"
          f" stationary={len(st)} saddles={len(act)}"
          f"  L2>0 in {sum(1 for r in act if r['L2'] > 0)}/{len(act)}"
          f"  viol={sum(r['viol'] for r in act)}  max| |v|-1 |={max([r['ue'] for r in act], default=0):.1e}")
    if not act:
        return
    rr = np.array([r["dip_meas"] / r["dip"] if r["dip"] == r["dip"] and r["dip"] != 0 else np.nan
                   for r in act])
    tr = np.array([r["t_meas"] / r["tp"] if r["tp"] == r["tp"] and r["tp"] > 0 else np.nan for r in act])
    c1 = np.array([r["c1"] for r in act])
    lo, hi = c1 < np.median(c1), c1 >= np.median(c1)
    print(f"{'':>28}  dip ratio meas/pred: all med={np.nanmedian(rr):9.6f}"
          f" [small c1: {np.nanmedian(rr[lo]):9.6f} | large c1: {np.nanmedian(rr[hi]):9.6f}]"
          f"  range=[{np.nanmin(rr):8.5f},{np.nanmax(rr):8.5f}]"
          f"  max|ratio-1|={np.nanmax(np.abs(rr - 1.0)):.2e}")
    print(f"{'':>28}  t* ratio: med={np.nanmedian(tr):9.6f} range=[{np.nanmin(tr):8.5f},"
          f"{np.nanmax(tr):8.5f}]   |c1| range=[{c1.min():.2e},{c1.max():.2e}]"
          f"   origin |lam(v1)-cf1| max={max(r['origin_err'] for r in act):.1e}")
    dm = np.array([r["dip_meas"] for r in act])
    nd = np.array([r["net"] for r in act])
    gp = np.array([r["gap"] for r in act])
    fl = np.array([r["floor"] for r in act])
    worse = nd < dm - fl                      # net strictly below the whole steepest ray
    print(f"{'':>28}  SIZE: med|dip_meas|={np.median(np.abs(dm)):8.2e} "
          f"med dip/gap={np.median(dm / gp):7.4f} (b48 net: 0.07-0.50) "
          f"min dip/gap={np.min(dm / gp):7.4f} max={np.max(dm / gp):7.4f}")
    print(f"{'':>28}  [E] net vs ray: net beats ray in {int(np.sum(worse))}/{len(act)} draws"
          f" (by >floor); med ray/net = {np.median(np.abs(dm) / np.maximum(np.abs(nd), 1e-300)):6.3f}"
          f"  med|net dip|={np.median(np.abs(nd)):8.2e}"
          f"  formula-predicted-gain > gap (impossible) in {sum(r['over'] for r in act)} draws")


def mixgen(cs, delta):
    def g():
        p = 6
        K0 = blocks(np.eye(p), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = (U[:, 0] + U[:, 1]) / np.sqrt(2.0)
        Vv = np.column_stack([np.linalg.eigh(np.eye(p) - np.outer(weak, weak))[1][:, 1:p], weak])
        ev = np.ones(p); ev[-1] = 1.0 - delta
        Lam = Vv @ np.diag(ev) @ Vv.T
        evk = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
        return Lam, K0 * min(1.0, cs[0] / np.sqrt(evk[-1]))
    return g


print("=" * 118)
print("[A-E] two-jet prediction of the steepest-ray dip: dip_pred = -L1^2/(4 L2), L1 = -F_C c1/F_lam.")
print("    Predicted ratios meas/pred -> 1 (dip) and t_meas/t_pred -> 1, with the dip ratio below 1")
print("    for large c1 (u is steepest only to first order).  FALSIFIER: a ratio near 0.7 or 1.3 at")
print("    small c1 -- then the implicit second derivative is wrong.")
for name, cs in SPECTRA.items():
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 118)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  A 小 |c1| 处 dip 比值应在 1 附近（[0.9,1.1]）；大 |c1| 处允许偏低（u 只一阶最陡）。")
print("     若小 c1 处比值≈0.7 或 1.3 => 我的隐函数二阶导账目错，公式作废。")
print("  B t* 比值应 -> 1；这是更锐的检验（不含小量平方）。")
print("  C L2<=0 的抽样不参与判读（射线上无内点极小），但要报告比例。")
print("  D 射线上 viol 必须为 0，否则 prop:chord 重开。")
print("  E 与 b48 的网对照：网里的真实竞争者若系统性低于本射线，说明 u 不是最优方向，")
print("     公式低估可得收益 —— 这正是我判断'要不要继续追家族最优'所需的信息。")
print("     本文件在同一批抽样上重跑网（64 方向 x 64 半径，含该射线），可逐抽样比较。")
print("  F dip_pred 不得大于存在的收益 gap=cf1-lam_min(G)（prop:chord 的下界），违反即账目错。")
