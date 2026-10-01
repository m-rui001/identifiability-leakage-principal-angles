"""b61 -- is (beat, angle) one law or two?  Parameter-free Taylor along the geodesic v_best -> v*.

Why (registered in b60's 判读 and community.md 52.5): the sphere minimum beats lambda_1(v*) by a
quantity that spans three orders of magnitude across cells (beat/floor 1.3e3 ... 1.7e6) while the
ANGLE stays in [0, 1.05e-5] rad.  So "how far the direction moved" and "how much lambda moved" are
not obviously the same measurement.  b60's [B-CLAIM-62] says they are NOT controlled by one quantity.

The test is a two-term expansion with BOTH coefficients measured locally at v_best, no fitting:
    u   = unit vector perpendicular to v_best pointing at v*    (so v* = cos(th)v_best + sin(th)u)
    d1  = grad_sphere(v_best) . u          (first derivative along the geodesic)
    d2  = kappa   = d^2 lambda_1 / d h^2 along that geodesic at h=0
    pred(th) = d1*th + 0.5*d2*th^2       vs   beat = lambda_1(v*) - lambda_1(v_best)  (measured)
Every coefficient gets a SECOND, independent route (values vs gradients), so each has a measured
floor -- rule 16: no band may be quoted below the disagreement of two real routes.

GATES (failure voids the round):
  G1  b60's exact-vs-exact gradient identity re-checked here (grad.v' vs (1+x^2) dlam_exact) <=1e-8.
  G2  evaluator on the geodesic points matches np.roots on the same quadratic <=1e-13.

FALSIFIERS (directions registered before the numbers exist):
  K1  d2 < -10*(d2's own floor): NEGATIVE curvature at the point my optimiser reported as a minimum.
      Then "v_best is a minimum" is void and every law statement must be relabelled.
  K2  |beat - pred| > 10*(measured floor of pred): the two-term expansion FAILS. lambda_1 is not C^2
      along the geodesic, or grad_sphere is inconsistent with the values -- the whole b60 machinery
      would be suspect, not just this claim.
  K3  |d1|*th > 0.1*beat: the LINEAR term is not negligible, i.e. v_best is not close to critical.
      Those draws are EXCLUDED from the pure-quadratic test and counted, not guessed away.
  K4  th ~ 0 but beat > 10*floor: same direction, different value -- an outright contradiction.
  K5  (the claim itself) ratio = beat/(0.5*d2*th^2) on the UNCONFOUNDED draws: if its per-cell medians
      agree to <20% relative AND the per-cell log beat vs log th slope is ~2, then ONE printable
      quantity (the curvature d2) controls both beat and angle, and [B-CLAIM-62] is REFUTED -- I will
      write it up as a corollary, not as a claim.
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ns = {}
_src = open("b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)
_ns["rng"] = np.random.default_rng(61201)          # own stream, distinct from b60's 61200

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
setup = _ns["setup"]
ray_L = _ns["ray_L"]
scalars = _ns["scalars"]
stats = _ns["stats"]
crossings = _ns["crossings"]
SPECTRA = _ns["SPECTRA"]
rng = _ns["rng"]

_bns = {}
_bns["__file__"] = "b59_setlevel_gridfree.py"
_b59 = open("b59_setlevel_gridfree.py", encoding="utf-8").read()
exec(compile(_b59.split("# ---------------------------------------------------------------- cell driver")[0],
             "b59-head", "exec"), _bns)
crossings_stable = _bns["crossings_stable"]
dlam_exact = _bns["dlam_exact"]


def lam_of_v(s, v):
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    return float(crossings_stable(np.array([B]), np.array([C]))[0]), B, C


def lam_roots(B, C):
    r = np.roots([C, -(B + C), B * (1.0 - B)])
    ok = r[np.abs(r.imag) < 1e-9].real
    ok = ok[(ok > 0.0) & (ok < 1.0)]
    return float(np.min(ok)) if ok.size else np.nan


def grad_sphere(s, v):
    lam, B, C = lam_of_v(s, v)
    disc = (B + C) ** 2 - 4.0 * C * B * (1.0 - B)
    Fl = 2.0 * C * lam - B - C
    if abs(Fl) < 1e-14:
        return lam, np.zeros_like(v), B, C, np.sqrt(max(disc, 0.0))
    lam_B = -(1.0 - lam - 2.0 * B) / Fl
    lam_C = -(lam * lam - lam) / Fl
    g = 2.0 * lam_B * (s["A0"] @ v) + 2.0 * lam_C * (s["B2"] @ v)
    return lam, g - v * float(v @ g), B, C, np.sqrt(max(disc, 0.0))


def sphere_descent(s, v0, iters=800):
    v = v0 / np.linalg.norm(v0)
    lam = lam_of_v(s, v)[0]
    for _ in range(iters):
        _lam, gs = grad_sphere(s, v)[:2]
        n = float(np.linalg.norm(gs))
        if not np.isfinite(lam) or n <= 0.0:
            break
        improved = False
        for t in (2.0, 0.5, 0.2, 0.05, 0.01, 2e-3, 1e-4, 1e-6, 1e-9):
            w = v - t * gs
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                continue
            w /= nw
            lw = lam_of_v(s, w)[0]
            if np.isfinite(lw) and lw < lam:
                tiny = lam - lw < 1e-17 * max(abs(lam), 1e-30)
                v, lam, improved = w, lw, True
                if tiny:
                    return float(lam), v, float(np.linalg.norm(grad_sphere(s, v)[1]))
                break
        if not improved:
            break
    return float(lam), v, float(np.linalg.norm(grad_sphere(s, v)[1]))


def geo(v, u, h):
    return np.cos(h) * v + np.sin(h) * u


def d2_value(s, v, u, h):
    """curvature by VALUES: (lam(h)-2lam(0)+lam(-h))/h^2, Richardson with h and h/2."""
    l0 = lam_of_v(s, v)[0]
    def k(hw):
        return (lam_of_v(s, geo(v, u, hw))[0] - 2.0 * l0 + lam_of_v(s, geo(v, u, -hw))[0]) / (hw * hw)
    k1, k2 = k(h), k(h / 2.0)
    return (4.0 * k2 - k1) / 3.0, abs(k1 - k2)


def d1_grad(s, v, u):
    """first derivative along the geodesic: GRADIENT route vs a Richardson-corrected VALUE route."""
    dg = float(grad_sphere(s, v)[1] @ u)
    def dv(hw):
        return (lam_of_v(s, geo(v, u, hw))[0] - lam_of_v(s, geo(v, u, -hw))[0]) / (2.0 * hw)
    v1, v2 = dv(HREF), dv(HREF / 2.0)
    dvr = (4.0 * v2 - v1) / 3.0                      # O(h^2) truncation removed
    return dg, dvr, max(abs(dg - dvr), abs(v1 - v2) / 3.0)


def d2_grad(s, v, u, h=1e-2):
    """curvature by the GRADIENT route: (gs(h).u - gs(-h).u)/(2h) -- independent of the value route."""
    a = float(grad_sphere(s, geo(v, u, h))[1] @ u)
    b = float(grad_sphere(s, geo(v, u, -h))[1] @ u)
    return (a - b) / (2.0 * h)


CELLS = [
    ("q=3 spread 5-75", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 5-75"])))(lam_spd(6))),
    ("q=3 spread 2-45-88", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))(lam_spd(6))),
    ("q=3 spread 30-88-89", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 30-88-89"])))(lam_spd(6))),
    ("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)),
                                                          np.cos(np.deg2rad(40))])))(lam_spd(5))),
    ("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(t))
                                                           for t in (5, 25, 55, 80)])))(lam_spd(7))),
]

g1_err, g2_err = [], []
rows = {tag: [] for tag, _ in CELLS}
HREF = 1e-2

for tag, gen in CELLS:
    for _d in range(10):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        gap = s["cf1"] - s["true"]
        uu = ray_L(s)
        if uu is not None:                                   # G1, same check as b60
            u = uu[0]
            sc = scalars(s, u)
            for x in np.geomspace(1e-4, 1.0, 12):
                th = np.arctan(x)
                v = np.cos(th) * s["v1"] + np.sin(th) * u
                dv = -np.sin(th) * s["v1"] + np.cos(th) * u
                gs = grad_sphere(s, v)[1]
                lam0, dl = dlam_exact(s, sc, x)
                if np.isfinite(lam0) and np.isfinite(dl) and abs(lam0) > 0.0:
                    g1_err.append(abs(float(gs @ dv) - (1.0 + x * x) * dl) / (abs(lam0) / max(x, 1e-12)))
        starts = {"v1": s["v1"], "vstar": s["vstar"], "eigB2": np.linalg.eigh(s["B2"])[1][:, -1],
                  "r1": rng.normal(size=s["A0"].shape[0]),
                  "r2": rng.normal(size=s["A0"].shape[0]),
                  "r3": rng.normal(size=s["A0"].shape[0])}
        res = {k: sphere_descent(s, w0) for k, w0 in starts.items()}
        best_key = min(res, key=lambda k: res[k][0])
        lam_best, v_best, gn = res[best_key]
        # evaluator floor on the geodesic (G2 + the rounding scale used in pred's floor)
        hs = np.linspace(0.0, 2.0 * HREF, 41)
        w0 = s["vstar"] - v_best * float(v_best @ s["vstar"])
        n0 = float(np.linalg.norm(w0))
        if n0 == 0.0:                      # v* IS v_best: any perpendicular direction will do
            w0 = rng.normal(size=s["A0"].shape[0])
            w0 = w0 - v_best * float(v_best @ w0)
            n0 = float(np.linalg.norm(w0))
        u0 = w0 / n0
        gv = np.array([geo(v_best, u0, h) for h in hs])
        gv = gv / np.linalg.norm(gv, axis=1, keepdims=True)
        Bgv = np.einsum("ij,jk,ik->i", gv, s["A0"], gv)
        Cgv = np.einsum("ij,jk,ik->i", gv, s["B2"], gv)
        lgv = crossings_stable(Bgv, Cgv)
        fl_g = float(np.nanmax(np.abs(lgv - crossings(Bgv, Cgv))))
        alt = lam_roots(float(v_best @ s["A0"] @ v_best), float(v_best @ s["B2"] @ v_best))
        if np.isfinite(alt):
            g2_err.append(abs(lam_best - alt))
        # the geodesic from v_best to v*
        c = min(1.0, abs(float(v_best @ s["vstar"])))
        th = float(np.arccos(c))
        w = s["vstar"] - v_best * float(v_best @ s["vstar"])
        nw = float(np.linalg.norm(w))
        u_dir = w / nw if nw > 0.0 else None
        lam_star = lam_of_v(s, s["vstar"])[0]
        beat = lam_star - lam_best
        d1g, d1v, fl_d1 = d1_grad(s, v_best, u_dir) if u_dir is not None else (np.nan,) * 3
        d2v, fl_d2v = d2_value(s, v_best, u_dir, HREF) if u_dir is not None else (np.nan, np.nan)
        d2g = d2_grad(s, v_best, u_dir) if u_dir is not None else np.nan
        fl_d2 = max(abs(d2v - d2g), fl_d2v)                  # disagreement of the two CURVATURE routes
        pred = d1g * th + 0.5 * d2v * th * th
        fl_pred = fl_d1 * th + 0.5 * fl_d2 * th * th + 2.0 * fl_g + 2.0 * np.finfo(float).eps * abs(lam_best)
        pred_q = 0.5 * d2v * th * th                          # pure quadratic version (K5)
        fterms = (fl_d1 * th, 0.5 * fl_d2 * th * th, 2.0 * fl_g,
                  2.0 * np.finfo(float).eps * abs(lam_best))
        # A LAW IS ONLY TESTABLE WHERE ITS PREDICTION STANDS ABOVE ITS OWN FLOOR.  fl_pred is the
        # sum of the measured two-route disagreements; if 0.5|d2|th^2 <= 10*fl_pred the test has no
        # leverage and the draw is reported as NOT INFORMATIVE, not as a pass (rule 16).
        informative = (0.5 * abs(d2v) * th * th) > 10.0 * fl_pred
        rows[tag].append(dict(gap=gap, th=th, beat=beat / gap, beat_lam=beat,
                              d1=d1g, d1v=d1v, fl_d1=fl_d1, d2=d2v, d2g=d2g, fl_d2=fl_d2,
                              pred=pred / gap, resid=(beat - pred) / gap, fl_pred=fl_pred / gap,
                              ft=[x / gap for x in fterms], info=informative,
                              ratio=(beat / pred_q) if pred_q > 0 else np.nan,
                              lin_share=(abs(d1g) * th / beat) if beat > 0 else np.nan,
                              L=gn, fl_g=fl_g, alpha=float(np.arccos(min(1.0, abs(
                                  float(s["vstar"] @ s["v1"]))))), key=best_key))

print("=" * 132)
print("[S0] gates")
print("=" * 132)
print(f"  G1 |grad.v' - (1+x^2)dlam_exact| / (|lam|/x) = {max(g1_err):.3e}  (<=1e-8)  n={len(g1_err)}"
      f"  -> {'PASS' if max(g1_err) <= 1e-8 else 'VOID: gradient wrong'}")
print(f"  G2 |lam_-(rationalised) - np.roots| at v_best = {max(g2_err):.3e}  (<=1e-13)"
      f"  -> {'PASS' if max(g2_err) <= 1e-13 else 'VOID: evaluator wrong'}")

print("\n" + "=" * 132)
print("[S1] per-draw local data of the expansion  beat = d1*th + 0.5*d2*th^2  (units: gap for beat/pred,")
print("     lambda for d1,d2;  fl_* = measured disagreement of the two independent routes)")
print("=" * 132)
print(f"  {'cell':>22} {'n':>3} {'th med':>9} {'th max':>9} {'beat med/gap':>12} {'|d1| med':>10} "
      f"{'d2 med':>10} {'d2/d2g med':>11} {'ratio med':>10} {'ratio spread':>12} {'K3':>3} {'K4':>3}")
for tag, _ in CELLS:
    a = rows[tag]
    if not a:
        continue
    th = np.array([r["th"] for r in a])
    bt = np.array([r["beat"] for r in a])
    rt = np.array([r["ratio"] for r in a], dtype=float)
    fin = rt[np.isfinite(rt)]
    k3 = sum(1 for r in a if np.isfinite(r["lin_share"]) and r["lin_share"] > 0.1)
    k4 = sum(1 for r in a if r["th"] < 1e-12 and r["beat_lam"] > 10.0 * r["fl_g"])
    print(f"  {tag:>22} {len(a):>3} {np.median(th):>9.2e} {th.max():>9.2e} {np.median(bt):>12.3e} "
          f"{np.median([abs(r['d1']) for r in a]):>10.2e} {np.median([r['d2'] for r in a]):>10.2e} "
          f"{np.median([r['d2'] / r['d2g'] for r in a if r['d2g'] != 0.0]):>11.4f} "
          f"{(float(np.median(fin)) if fin.size else float('nan')):>10.4f} "
          f"{(float(fin.max() / fin.min()) if fin.size > 1 and fin.min() > 0 else float('nan')):>12.3f} "
          f"{k3:>3} {k4:>3}")

print("\n" + "=" * 132)
print("[S2] the falsifiers")
print("=" * 132)
allr = [x for a in rows.values() for x in a]
k1 = [(r["d2"], r["fl_d2"]) for r in allr if np.isfinite(r["d2"]) and r["d2"] < -10.0 * r["fl_d2"]]
print(f"  K1 #(d2 < -10*d2's own floor) = {len(k1)}"
      + (f"   worst d2 = {min(x[0] for x in k1):.3e}" if k1 else "")
      + "   (negative curvature at a point my optimiser reported as a minimum -> 'minimum' void)")
res_v = np.array([r["resid"] for r in allr], dtype=float)
flp = np.array([r["fl_pred"] for r in allr], dtype=float)
inf = np.array([r["info"] for r in allr], dtype=bool)
bad = inf & np.isfinite(res_v) & (np.abs(res_v) > 10.0 * flp)
print(f"  K2 #(|beat - pred| > 10*floor(pred)) = {int(bad.sum())}/{int(np.isfinite(res_v).sum())}"
      f"   max |resid|/floor = {float(np.nanmax(np.abs(res_v) / flp)):.3e}"
      "   (two-term expansion along the geodesic)")
print(f"  K2-informativeness (rule 16): the quadratic term 0.5|d2|th^2 exceeds 10*floor(pred) on "
      f"{int(inf.sum())}/{len(allr)} draws; on the other {len(allr) - int(inf.sum())} the test has NO "
      f"leverage and is not counted as a pass"
      f"   (max |resid|/floor over informative draws = "
      f"{float(np.nanmax(np.abs(res_v[inf]) / flp[inf])) if inf.any() else float('nan'):.3e})")
lin = np.array([r["lin_share"] for r in allr], dtype=float)
print(f"  K3 #(|d1|*th > 0.1*beat, linear term not negligible) = {int(np.nansum(lin > 0.1))}"
      f"   med lin share = {float(np.nanmedian(lin)):.3e}")
k4 = sum(1 for r in allr if r["th"] < 1e-12 and r["beat_lam"] > 10.0 * r["fl_g"])
print(f"  K4 #(th<1e-12 but beat > 10 floors) = {k4}   (same direction, different value: contradiction)")
print(f"  #(th == 0 exactly) = {sum(1 for r in allr if r['th'] == 0.0)}"
      f"   #(beat <= 0) = {sum(1 for r in allr if r['beat_lam'] <= 0.0)}")
# K5: the claim's own test, on UNCONFOUNDED draws only
sel = [r for r in allr if np.isfinite(r["ratio"]) and r["th"] > 1e-9
       and (not np.isfinite(r["lin_share"]) or r["lin_share"] <= 0.1) and r["beat"] > 10.0 * r["fl_g"] / r["gap"]]
sel_i = [r for r in sel if r["info"]]
print(f"  K5 testable draws (th>1e-9, linear term negligible, beat>10 floors) = {len(sel)}"
      f"   of which INFORMATIVE (prediction above its own floor) = {len(sel_i)}")
for nm, group in (("all testable", sel), ("informative only", sel_i)):
    if not group:
        print(f"     {nm}: (empty)")
        continue
    rr = np.array([r["ratio"] for r in group])
    dev = float(np.median(np.abs(rr - 1.0)))
    verdict = "REFUTED: one quantity (d2) controls beat and angle" if dev < 0.2 else "SURVIVES"
    print(f"     {nm}: ratio beat/(0.5*d2*th^2) med={np.median(rr):.4f} min={rr.min():.4f} "
          f"max={rr.max():.4f}  med|ratio-1|={dev:.4f}   -> [B-CLAIM-62(i)] {verdict}")
lg = [(np.log(r["beat"]), np.log(r["th"])) for r in sel_i if r["beat"] > 0 and r["th"] > 0]
if len(lg) >= 3:
    x = np.array([a[1] for a in lg]); y = np.array([a[0] for a in lg])
    slope = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])
    print(f"     pooled log beat vs log th on INFORMATIVE draws = {slope:.3f}"
          f" (2.000 quadratic, 1.000 linear), n={len(lg)}")
print(f"  [context] |d2| med = {np.median([abs(r['d2']) for r in allr]):.3e}"
      f"  min = {min(abs(r['d2']) for r in allr):.3e}  max = {max(abs(r['d2']) for r in allr):.3e}")
print(f"  [context] alpha=angle(v*,v1): med={np.median([r['alpha'] for r in allr]):.4f}"
      f" min={min(r['alpha'] for r in allr):.4f} max={max(r['alpha'] for r in allr):.4f}")
print(f"  [context] best start counts: { {k: sum(1 for r in allr if r['key'] == k) for k in ('v1','vstar','eigB2','r1','r2','r3')} }")

print("\n" + "=" * 132)
print("[S3] DIAGNOSTIC for the tension between K5 (per-draw ratio ~ 1) and the pooled slope (~0.57).")
print("     Pooled log beat vs log th mixes CELLS; if d2 differs between cells the pooled slope is")
print("     not the law's exponent.  Printed per cell: the within-cell slope, and every testable draw.")
print("=" * 132)
print(f"  {'cell':>22} {'n':>3} {'th range':>22} {'beat/gap range':>22} {'d2 range':>18} "
      f"{'ratio med':>9} {'within-cell slope':>18} {'info':>5}")
for tag, _ in CELLS:
    a = [r for r in rows[tag] if r["th"] > 1e-9 and r["beat"] > 0 and np.isfinite(r["ratio"])]
    if not a:
        print(f"  {tag:>22} {len(a):>3}   (no testable draw: th=0 or beat below floor)")
        continue
    th = np.array([r["th"] for r in a]); bt = np.array([r["beat"] for r in a])
    d2 = np.array([r["d2"] for r in a]); rt = np.array([r["ratio"] for r in a])
    sl = float(np.polyfit(np.log(th) - np.log(th).mean(),
                          np.log(bt) - np.log(bt).mean(), 1)[0]) if len(a) >= 3 else float("nan")
    ni = sum(1 for r in a if r["info"])
    print(f"  {tag:>22} {len(a):>3} [{th.min():.2e},{th.max():.2e}] [{bt.min():.2e},{bt.max():.2e}]"
          f" [{d2.min():.2e},{d2.max():.2e}] {np.median(rt):>9.4f} {sl:>18.3f} {ni:>5}")
print(f"  {'draw':>6} {'cell':>22} {'th':>10} {'beat/gap':>10} {'0.5*d2*th^2':>12} {'ratio':>8} "
      f"{'resid/fl_pred':>13} {'lin share':>10} {'fl_d1*th':>10} {'0.5*fl_d2*th^2':>15} {'2*fl_g':>10} {'info':>5}")
i = 0
for tag, _ in CELLS:
    for r in rows[tag]:
        i += 1
        if r["th"] <= 1e-9 or not np.isfinite(r["ratio"]):
            continue
        print(f"  {i:>6} {tag:>22} {r['th']:>10.3e} {r['beat']:>10.3e} "
              f"{0.5 * r['d2'] * r['th'] ** 2 / r['gap']:>12.3e} {r['ratio']:>8.4f} "
              f"{r['resid'] / r['fl_pred'] if r['fl_pred'] > 0 else float('nan'):>13.3e} "
              f"{r['lin_share']:>10.3e} {r['ft'][0]:>10.3e} {r['ft'][1]:>15.3e} {r['ft'][2]:>10.3e}"
              f" {'yes' if r['info'] else 'no':>5}")


print("\n" + "=" * 132)
print("判读（预登记，跑前写下的方向；跑后只填数，不改方向）")
print("  K2 是本轮的真门禁：`pred` 只用 `v_best` 一点的两条独立路线（值 vs 梯度）测出 d1、d2，")
print("     却要对整个 2e-5 弧度的位移负责。|beat-pred| 若在 10 倍地板内为 0，则 `λ₁` 沿测地线是 C²，")
print("     且 b60 的梯度与函数值互相自洽（这是比 G1 更强的跨路线检验，因为它用到了二阶）。")
print("     K2 大量触发 => 我的梯度公式或 λ₋ 选支在高阶处不一致，那么 open item 1 的球面结论要重新审视。")
print("  K1 触发 => 「球面最小点」这个词作废，正文 (viii) 里所有 'sphere minimum' 要降级成")
print("     「我的优化器报告的最优点」，并计入撤回清单。")
print("  K3 是混淆项计数，不是失败：`v_best` 的 |grad| 在 b60 里就有 19/50 超过 1e-6|λ|，")
print("     所以线性项 d1*th 可能与 0.5*d2*th² 同阶。**这些抽样必须从 K5 剔除**，否则我拿一个")
print("     一次项主导的数去验二次律，是量具在骗我。")
print("  K5 才是 [B-CLAIM-62] 的判决：ratio 若集中在 1（±20%），说明 beat 与角度由**同一个可打印量")
print("     （`λ₁` 在 `v_best` 处沿测地线的球面曲率 d2）**控制 —— 那我的 claim 被自己的数据否掉，")
print("     按 (i) 写成本节推论；若 ratio 跨格散布 >20%，claim 存活，剩下的问题是「什么控制 d2」。")
print("  斜率 2 => 二次律；斜率 1 => 一次项（即 v_best 未达驻点）主导，本轮的 claim 检验无效。")
print("  不注册：`beat` 的绝对大小、`deficit`（b60 已给）。")

print("\n" + "=" * 132)
print("REGISTRATION HISTORY (v1 -> v2), written after the first run, and it makes the test HARDER:")
print("  v1's floor for the linear coefficient was |d1_gradient - d1_value(h=1e-2)|, which is the")
print("     O(h^2) TRUNCATION of the value stencil, not evaluator noise. At th~3e-7 that inflated term")
print("     (fl_d1*th up to 4.1e-6 of gap at th=2.6e-8) was 10-400x LARGER than the quantity being")
print("     tested (0.5*d2*th^2 down to 9.2e-9 of gap), so K2's '0 violations' was vacuous -- a band")
print("     wider than the effect.  v2 Richardson-")
print("     corrects d1 (h and h/2), and adds the informative gate: a draw counts only if the")
print("     prediction stands 10x above its own floor. Rule 16's failure mode again: the measuring")
print("     stick changed, the goal did not -- and the change can only REMOVE passes, never add them.")

