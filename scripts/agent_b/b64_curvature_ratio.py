"""b64 -- [B-CLAIM-65]: does b61's quadratic law beat = 1/2 kappa theta^2 hold PER DRAW in the
near-coincident top-cosine design (b63b's DEGEN family)?

ORIGIN (registered in community.md 55.8).  b61 measured the per-draw ratio at median 1.0000 but only on
10/50 draws with leverage, and b63b's DEGEN family (top two cosines 0.90 and 0.90-1e-4) has theta's
MEDIAN 23.5x larger than SPACED.  So the near-coincident regime is where the law has the most chance of
being testable -- and it is the regime b61 never swept.  This script is that test, no new design.

MACHINERY IS b61's, VERBATIM (exec'd from b61_geodesic_law.py, only the rng stream is re-bound):
    u   = unit vector at v_b pointing at v*     (v* = cos(th) v_b + sin(th) u)
    d1  = grad_sphere(v_b) . u                  (gradient route, + Richardson value route)
    k   = d^2 lambda_1 / dh^2 along that geodesic at h=0, measured TWICE: values (Richardson) and
          gradients.  kappa's two-route disagreement is its floor.
    pred_q = 0.5*k*th^2                          vs   beat = lambda_1(v*) - lambda_1(v_b)

ORDER OF THE REPORT IS FORCED BY 55.8: kappa's two-route agreement FIRST (a claim about a ratio whose
denominator I do not trust is not a claim), then the leverage count with the registered stop condition,
then the per-draw ratio.

PRE-RUN CONSIDERATION, written before any number existed (it only makes the test harder, per rule 17):
    The band [0.98, 1.02] is a +-2% statement about pred_q.  A draw whose prediction floor is, say, 10%
    of pred_q CANNOT resolve +-2%: the measuring stick is 5x the effect, so "inside the band" there is
    vacuous and "outside the band" is not yet a falsification.  So besides 55.8's registered 10x gate I
    print a second count, band-resolvable = (pred_q > 100*floor(pred)), and the falsifier is stated in
    the floor's own units: |beat - pred_q| > floor(pred) is a REAL deviation, anything smaller is not a
    test.  The stop condition of 55.8 (<20 of 40 with leverage) is unchanged and is the one that gates
    the write-up.

DESIGN: DEGEN cosines (0.90, 0.90-1e-4, 0.40), p from lam_spd(6), 40 draws, b61's exact 6-start set.

GATES (a failure voids the round, it does not weaken the claim):
  G1  the geodesic endpoint identity: |cos(th) v_b + sin(th) u - v*|_inf <= 1e-12 AND |u|=1, u . v_b=0.
      This is the construction that the whole expansion is written on; if it fails, th and u are not
      what the docstring says (rule 9).
  G2  evaluator: |lam_of_v(v) - np.roots(B,C)| <= 1e-13 at both v_b and v* (b60/b61's check).
  G3  b61's exact-vs-exact gradient identity G1 on the ray (re-run here, 12 x values, <=1e-8).

PRECONDITION P1 (55.8, reported first): median |k_values/k_grad - 1| <= 0.05 and max <= 0.25 over the
      testable draws.  If it fails, kappa is not a measurable quantity, [B-CLAIM-65] is VOID, and I say
      so instead of quoting the ratio.

FALSIFIERS for the claim itself (directions written before the numbers exist):
  C1  any testable draw with |beat - pred_q| > floor(pred): the two-term expansion fails in the
      near-coincident regime.  (K2 of b61 was 0/50 there; this is the regime b61 never reached.)
  C2  #(with leverage) < 20  =>  STOP CONDITION, not a result: "the law is untested near coincidence"
      goes into open item 1 and b61 is NOT extended.
  C3  median ratio outside [0.98, 1.02] on band-resolvable draws: the law holds in order but not in
      constant, so the chapter's "parameter-free" wording would have to go.
  C4  the linear share |d1|*th / beat > 0.1 on a draw: v_b is not close enough to critical for a pure
      quadratic test; those draws are EXCLUDED and counted (b61's K3), never averaged away.

NOT REGISTERED (explicitly, because 55.5/55.8 forbid them as evidence): NO pooled log-log slope is used
as evidence for or against the law here (55.5 item 4; a pooled slope mixes the exponent with kappa's own
dependence on th); NO correlation of the ratio against within-group rho_A, which b63b showed cannot vary
by construction (55.5 item 2).  Both are printed only as diagnostics, labelled as such.
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ns = {}
_src = open("b61_geodesic_law.py", encoding="utf-8").read()
exec(compile(_src.split("CELLS = [")[0], "b61-head", "exec"), _ns)
# 55.8 reuses b61's helpers, which close over the exec'd namespace's rng.  Re-bind ONE stream here so
# this round's draws are reproducible and are not b61's (61201) or b63b's (61204).
_ns["rng"] = np.random.default_rng(61206)

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
setup = _ns["setup"]
ray_L = _ns["ray_L"]
scalars = _ns["scalars"]
dlam_exact = _ns["dlam_exact"]
crossings = _ns["crossings"]
crossings_stable = _ns["crossings_stable"]
lam_of_v = _ns["lam_of_v"]
lam_roots = _ns["lam_roots"]
grad_sphere = _ns["grad_sphere"]
sphere_descent = _ns["sphere_descent"]
geo = _ns["geo"]
d1_grad = _ns["d1_grad"]
d2_value = _ns["d2_value"]
d2_grad = _ns["d2_grad"]
rng = _ns["rng"]
# HREF is set in b61 AFTER its "CELLS = [" line, so the exec'd head does not carry it.  d1_grad closes
# over the head namespace and reads HREF at CALL time, so injecting the same value 1e-2 here reproduces
# b61's stencil exactly rather than silently changing the step.
_ns["HREF"] = 1e-2
HREF = _ns["HREF"]

COS = [0.90, 0.90 - 1e-4, 0.40]           # b63b's DEGEN spectrum, verbatim
NDRAW = 40
EPS = np.finfo(float).eps

g1, g2, g3, g1old, nneg = [], [], [], [], 0
rows = []
for _d in range(NDRAW):
    Lam = lam_spd(6)
    K = blocks(Lam, COS)
    s = setup(Lam, K)
    if s is None:
        continue
    gap = s["cf1"] - s["true"]
    # G3 -- b61's gradient-vs-exact-derivative identity on the ray through v1
    uu = ray_L(s)
    if uu is not None:
        u = uu[0]
        sc = scalars(s, u)
        for x in np.geomspace(1e-4, 1.0, 12):
            th = np.arctan(x)
            v = np.cos(th) * s["v1"] + np.sin(th) * u
            dv = -np.sin(th) * s["v1"] + np.cos(th) * u
            gs = grad_sphere(s, v)[1]
            lam0, dl = dlam_exact(s, sc, x)
            if np.isfinite(lam0) and np.isfinite(dl) and abs(lam0) > 0.0:
                g3.append(abs(float(gs @ dv) - (1.0 + x * x) * dl) / (abs(lam0) / max(x, 1e-12)))
    starts = [s["v1"], s["vstar"], np.linalg.eigh(s["B2"])[1][:, -1],
              rng.normal(size=s["A0"].shape[0]), rng.normal(size=s["A0"].shape[0]),
              rng.normal(size=s["A0"].shape[0])]
    lam_b, v_b, gn_b = min([sphere_descent(s, w0) for w0 in starts], key=lambda r: r[0])
    lam_star, Bv, Cv = lam_of_v(s, s["vstar"])
    a_b, Bb, Cb = lam_of_v(s, v_b)
    beat = lam_star - lam_b
    g2 += [abs(lam_star - lam_roots(Bv, Cv)), abs(a_b - lam_roots(Bb, Cb))]
    # the geodesic from v_b to v*  (G1: the construction the expansion is written on)
    # the geodesic from v_b to v*.  SIGN CONVENTION MATTERS and was my G1 failure: lambda_1(v) depends
    # on v only through the quadratic forms B(v), C(v), so it is INVARIANT under v -> -v, and the
    # descent returns whichever sign it met.  theta is therefore the angle to the LINE {+-v*}, and u
    # must point at sign(c)*v*, not at v*.
    c = float(v_b @ s["vstar"])
    sgn = 1.0 if c >= 0.0 else -1.0
    tgt = sgn * s["vstar"]
    th = float(np.arccos(min(1.0, abs(c))))
    w = tgt - v_b * float(v_b @ tgt)
    nw = float(np.linalg.norm(w))
    u_dir = w / nw if nw > 0.0 else None
    w_old = s["vstar"] - v_b * c                      # b60/b61/b63b's convention (signed projection)
    nw_old = float(np.linalg.norm(w_old))
    u_old = w_old / nw_old if nw_old > 0.0 else None
    if u_old is not None:
        g1old.append(float(np.max(np.abs(geo(v_b, u_old, th) - s["vstar"]))))
    if sgn < 0.0:
        nneg += 1
    if u_dir is not None:
        endpoint = geo(v_b, u_dir, th)
        g1.append(max(float(np.max(np.abs(endpoint - tgt))),
                      abs(float(np.linalg.norm(u_dir)) - 1.0),
                      abs(float(u_dir @ v_b))))
    d1g, d1v, fl_d1 = d1_grad(s, v_b, u_dir) if u_dir is not None else (np.nan,) * 3
    d2v, fl_d2v = d2_value(s, v_b, u_dir, HREF) if u_dir is not None else (np.nan, np.nan)
    d2g = d2_grad(s, v_b, u_dir, HREF) if u_dir is not None else np.nan
    # the sign defect's reach, measured rather than argued: re-do both routes with the OLD direction.
    # kappa's routes are symmetric stencils in h, so u -> -u must leave them unchanged; only d1 flips.
    d1o, _d1ov, _f1o = d1_grad(s, v_b, u_old) if u_old is not None else (np.nan,) * 3
    d2vo, _fo = d2_value(s, v_b, u_old, HREF) if u_old is not None else (np.nan, np.nan)
    d2go = d2_grad(s, v_b, u_old, HREF) if u_old is not None else np.nan
    # POST-RUN addition (see REGISTRATION HISTORY, consumes no rng draws): the third derivative along
    # the same geodesic, by the 4-point antisymmetric stencil, Richardson-corrected in h.
    def _f(hw, uu):
        return lam_of_v(s, geo(v_b, uu, hw))[0]
    def _d3(hw):
        return (_f(-2 * hw, u_dir) - 2 * _f(-hw, u_dir) + 2 * _f(hw, u_dir) - _f(2 * hw, u_dir)) / (2 * hw ** 3)
    a3, b3 = _d3(HREF), _d3(HREF / 2.0)
    d3v = (4.0 * b3 - a3) / 3.0
    fl_d3 = abs(a3 - b3) / 3.0
    # floor of kappa: the disagreement of the two ROUTES plus the value route's own Richardson spread
    fl_k = max(abs(d2v - d2g), fl_d2v)
    pred_q = 0.5 * d2v * th * th
    fl_pred = fl_d1 * th + 0.5 * fl_k * th * th + 2.0 * max(g2[-2], g2[-1]) + 2.0 * EPS * abs(lam_b)
    rows.append(dict(gap=gap, th=th, beat=beat, bg=beat / gap, gn=gn_b, d1=d1g, d1v=d1v, fl_d1=fl_d1,
                     d2v=d2v, d2g=d2g, fl_k=fl_k, pred_q=pred_q, fl_pred=fl_pred,
                     sgn=sgn, d1o=d1o, d2vo=d2vo, d2go=d2go, d3v=d3v, fl_d3=fl_d3,
                     pred3=d3v * th ** 3 / 6.0,
                     lam_b=lam_b, lam_star=lam_star, true=s["true"], cf1=s["cf1"]))

print("=" * 132)
print("[G] gates (a failure voids the round)")
print("=" * 132)
print(f"  G1 max(|cos(th)v_b+sin(th)u - sign(v_b.v*)v*|_inf, ||u||-1, |u.v_b|) = {max(g1):.3e}"
      f"  (<=1e-12) n={len(g1)}"
      f"  -> {'PASS' if max(g1) <= 1e-12 else 'VOID: th and u are not the geodesic I wrote'}")
print(f"  G2 max |lam_of_v - np.roots| over v_b and v* = {max(g2):.3e}  (<=1e-13)"
      f"  -> {'PASS' if max(g2) <= 1e-13 else 'VOID: evaluator'}")
print(f"  G3 b61 gradient identity, max rel defect = {max(g3):.3e}  (<=1e-8)  n={len(g3)}"
      f"  -> {'PASS' if max(g3) <= 1e-8 else 'VOID: grad_sphere inconsistent with values'}")

print("\n" + "=" * 132)
print("[S1] the G1 failure was a SIGN convention, measured not argued.  lambda_1(v) sees v only through")
print("     B(v), C(v), so lambda_1(-v) = lambda_1(v): the descent may return either sign, theta is the")
print("     angle to the LINE {+-v*}, and u must point at sign(v_b.v*)*v*.")
print("=" * 132)
print(f"  #draws with v_b.v* < 0 (descent returned the opposite sign of v*): {nneg}/{len(rows)}")
print(f"  max |geo(v_b, u_new, th) - sign*v*| = {max(g1):.3e}  (G1, PASS side)")
print(f"  max |geo(v_b, u_OLD, th) - v*|      = {max(g1old):.3e}  <- what the first run measured")
dk = np.array([abs(r["d2v"] - r["d2vo"]) for r in rows])
dg = np.array([abs(r["d2g"] - r["d2go"]) for r in rows])
dd1 = np.array([abs(r["d1"] + r["d1o"]) if r["sgn"] < 0 else abs(r["d1"] - r["d1o"]) for r in rows])
print(f"  reach of the defect: max |k_values(u_new) - k_values(u_old)| = {dk.max():.3e},"
      f"  |k_grad(u_new) - k_grad(u_old)| = {dg.max():.3e}")
print(f"                   max |d1(u_new) + d1(u_old)| on the sign-flipped draws = {dd1.max():.3e}"
      f"  (the stencils are symmetric in h, so u -> -u must flip d1 only -- as measured)")
rq = np.array([abs(r["beat"] / (0.5 * r["d2vo"] * r["th"] ** 2) - r["beat"] / r["pred_q"]) for r in rows
               if r["d2vo"] > 0 and r["th"] > 0])
print(f"  max |ratio_old_convention - ratio_new| over 40 draws = {rq.max():.3e}"
      f"  -> the registered ratio is UNAFFECTED (it uses only kappa and theta), so [B-CLAIM-65]'s"
      f" numbers below stand as measured; only my G1 sentence was wrong")

print("\n" + "=" * 132)
print("[P1] FIRST RESULT, BEFORE ANY RATIO: is kappa measurable at all?  (routes: Richardson values vs"
      "\n     gradients, both evaluated at v_b along the geodesic toward v*)")
print("=" * 132)
krel = np.array([abs(r["d2v"] / r["d2g"] - 1.0) for r in rows
                 if r["d2g"] != 0.0 and np.isfinite(r["d2g"])])
print(f"  n={len(krel)}   |k_v/k_g - 1|: med={np.median(krel):.3e} p90={np.quantile(krel, 0.9):.3e}"
      f" max={krel.max():.3e}")
print(f"  k_values: med={np.median([r['d2v'] for r in rows]):.4e}"
      f"  range=[{min(r['d2v'] for r in rows):.4e}, {max(r['d2v'] for r in rows):.4e}]"
      f"   sign: {sum(1 for r in rows if r['d2v'] > 0)} positive / {sum(1 for r in rows if r['d2v'] < 0)} negative")
print(f"  floor of k (= max(|k_v-k_g|, value-route Richardson spread)): med={np.median([r['fl_k'] for r in rows]):.3e}")
p1 = (len(krel) > 0) and (np.median(krel) <= 0.05) and (krel.max() <= 0.25)
print(f"  -> P1 {'PASS: kappa is a measured quantity here' if p1 else 'FAIL: kappa is NOT trustworthy, [B-CLAIM-65] VOID (C-rule: report, do not quote the ratio)'}")

# testable set: u exists, th resolvable, quadratic term dominant over the linear one (C4)
def testable(r):
    return (np.isfinite(r["d2v"]) and r["th"] > 1e-9 and r["beat"] > 0
            and (abs(r["d1"]) * r["th"] <= 0.1 * r["beat"]))
nex = sum(1 for r in rows if np.isfinite(r["d2v"]) and r["th"] > 1e-9 and r["beat"] > 0)
tex = [r for r in rows if testable(r)]
print(f"\n  C4 draws excluded because the LINEAR term |d1|th exceeds 0.1*beat: "
      f"{nex - len(tex)} of {nex} draws with th>1e-9 and beat>0")

lev = [r for r in tex if r["pred_q"] > 10.0 * r["fl_pred"]]           # 55.8's registered gate
bnd = [r for r in tex if r["pred_q"] > 100.0 * r["fl_pred"]]          # the +-2% band's own need
print("\n" + "=" * 132)
print("[P2] leverage (rule 17: a prediction below its own floor is NOT a test, never a pass)")
print("=" * 132)
print(f"  draws built = {len(rows)}/{NDRAW}   testable (th>1e-9, beat>0, linear term negligible) = {len(tex)}")
print(f"  with leverage (pred_q > 10*floor) = {len(lev)}/{len(tex)}"
      f"   -> registered STOP CONDITION (<20 of 40): {'NOT TRIGGERED' if len(lev) >= 20 else 'TRIGGERED -> b61 is NOT extended, and open item 1 records that the law is UNTESTED near coincidence'}")
print(f"  band-resolvable (pred_q > 100*floor, needed to read a +-2% band) = {len(bnd)}/{len(tex)}")
if tex:
    lr = np.array([r["pred_q"] / r["fl_pred"] for r in tex])
    print(f"  pred_q/floor over testable draws: min={lr.min():.3e} med={np.median(lr):.3e} max={lr.max():.3e}")

print("\n" + "=" * 132)
print("[P3] [B-CLAIM-65]: ratio = beat / (0.5*k*th^2) PER DRAW")
print("=" * 132)
for nm, group in (("with leverage (10x)", lev), ("band-resolvable (100x)", bnd)):
    if not group:
        print(f"  {nm}: (empty)")
        continue
    rt = np.array([r["beat"] / r["pred_q"] for r in group])
    dev = np.array([abs(r["beat"] - r["pred_q"]) / r["fl_pred"] for r in group])
    inband = int(((rt >= 0.98) & (rt <= 1.02)).sum())
    print(f"  {nm}: n={len(rt)}  ratio med={np.median(rt):.4f} min={rt.min():.4f} max={rt.max():.4f}"
          f"  #in[0.98,1.02]={inband}/{len(rt)}")
    print(f"      |beat-pred_q|/floor(pred): med={np.median(dev):.3e} max={dev.max():.3e}"
          f"  #>1 (=C1, a deviation above the floor is REAL) = {int((dev > 1).sum())}")
    if nm.startswith("band"):
        print(f"  -> C3 (band): {'ratio med INSIDE [0.98,1.02]' if 0.98 <= np.median(rt) <= 1.02 else 'ratio med OUTSIDE the band -- the parameter-free wording must go'}")
c1 = [r for r in tex if abs(r["beat"] - r["pred_q"]) > r["fl_pred"] and r["pred_q"] > 10.0 * r["fl_pred"]]
print(f"  C1 #( |beat - pred_q| > floor(pred) on draws with leverage ) = {len(c1)}/{len(lev)}")
print(f"  #(beat<=0) = {sum(1 for r in rows if r['beat'] <= 0)}"
      f"   #(th==0) = {sum(1 for r in rows if r['th'] == 0.0)}"
      f"   #(stationary v_b: |grad|<=1e-6|lam|) = {sum(1 for r in rows if r['gn'] <= 1e-6 * abs(r['lam_b']))}/{len(rows)}")

print("\n" + "=" * 132)
print("[P3b] POST-RUN, NOT REGISTERED -- what the residual  beat - pred_q  is.")
print(f"      C1 fired on {len(c1)}/{len(lev)} while the ratio stayed inside the +-2% band, so the")
print("      deviation is real but small; the natural candidate is the THIRD term of the same")
print("      geodesic Taylor series.")
print("=" * 132)
sub = [r for r in lev]
share = np.array([r["pred3"] / (r["beat"] - r["pred_q"]) for r in sub
                  if abs(r["beat"] - r["pred_q"]) > 0])
fl3 = np.array([r["fl_pred"] + r["fl_d3"] * r["th"] ** 3 / 6.0 for r in sub])
after = np.array([abs(r["beat"] - r["pred_q"] - r["pred3"]) for r in sub])
print(f"  d3 (values route, Richardson): med={np.median([r['d3v'] for r in sub]):.4e}"
      f"  range=[{min(r['d3v'] for r in sub):.4e}, {max(r['d3v'] for r in sub):.4e}]"
      f"   #(d3>0)={sum(1 for r in sub if r['d3v'] > 0)}/{len(sub)}")
print(f"  cubic term / quadratic term  = (1/6)d3 th^2 / (1/2 k) : med="
      f"{np.median([(r['d3v'] * r['th'] / 3.0) / r['d2v'] for r in sub]):.3e}"
      f"  max={max([(r['d3v'] * r['th'] / 3.0) / r['d2v'] for r in sub]):.3e}")
ncub = int(((share > 0.5) & (share < 2.0)).sum())
print(f"  pred3 / (beat - pred_q): med={np.median(share):.3e}  range=[{share.min():.3e},{share.max():.3e}]"
      f"   #in[0.5,2]={ncub}/{len(share)}")
print(f"  after removing the cubic term: #(|beat-pred_q-pred3| > floor+floor3) = "
      f"{int((after > fl3).sum())}/{len(sub)}")
rr = np.array([abs(r["beat"] / r["pred_q"] - 1.0) for r in sub])
print(f"  |ratio-1|: med={np.median(rr):.3e} max={rr.max():.3e}"
      f"   (theta range here {[f'{x:.2e}' for x in (min(r['th'] for r in sub), max(r['th'] for r in sub))]})")
if np.all(rr > 0) and np.all([r["th"] > 0 for r in sub]):
    thv = np.log([r["th"] for r in sub]); rv = np.log(rr)
    kspan = max(r["d2v"] for r in sub) / min(r["d2v"] for r in sub)
    print(f"  slope of log|ratio-1| vs log theta = {float(np.polyfit(thv - thv.mean(), rv - rv.mean(), 1)[0]):.3f}"
          f"   (1.0 would say the residual is the cubic term; NO exponent claim is made from a pooled"
          f" slope -- 55.5 item 4 -- and k itself spans {kspan:.0f}x in this family, which is the confound)")
# what DOES the residual track?  rule 19 first: print each regressor's range, a regressor that cannot
# vary has no correlation to read (this is the check I failed in b63b's S3).
ls = np.array([abs(r["d1"]) * r["th"] / r["beat"] for r in sub])
absg = np.array([abs(r["d1"]) for r in sub])
kgr = np.array([r["d2v"] for r in sub])
print(f"  regressor ranges (must be non-degenerate before any correlation is read):")
print(f"     linear share |d1|th/beat: [{ls.min():.3e},{ls.max():.3e}]  ratio max/min={ls.max()/ls.min():.1f}")
print(f"     |d1|                    : [{absg.min():.3e},{absg.max():.3e}]  ratio max/min={absg.max()/absg.min():.1f}")
print(f"     |grad(v_b)|             : [{min(r['gn'] for r in sub):.3e},{max(r['gn'] for r in sub):.3e}]")
print(f"     theta                   : [{min(r['th'] for r in sub):.3e},{max(r['th'] for r in sub):.3e}]")
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
print(f"  Spearman(|ratio-1|, .): linear share={spear(rr, ls):+.3f}   |d1|={spear(rr, absg):+.3f}"
      f"   theta={spear(rr, np.array([r['th'] for r in sub])):+.3f}   k={spear(rr, kgr):+.3f}")
order = np.argsort(-rr)
print(f"  the two largest |ratio-1| draws are #{order[0]+1}, #{order[1]+1};"
      f" dropping them: max|ratio-1|={rr[order[2:]].max():.3e}"
      f"  (a SENSITIVITY reading, not a re-gate -- the drop is by the size of the very quantity tested)")
print(f"  and those two draws' linear shares are {ls[order[0]]:.3e}, {ls[order[1]]:.3e} against a"
      f" median {np.median(ls):.3e}; #(|grad(v_b)|>1e-6|lam|) in the family = "
      f"{sum(1 for r in rows if r['gn'] > 1e-6 * abs(r['lam_b']))}")
# the linear term is not just A correlate, it is the PREDICTED correction:
#   beat = d1*th + 0.5*k*th^2  =>  beat/pred_q - 1 = 2*d1/(k*th)  exactly.  Check it per draw.
pf = np.array([r["d1"] * r["th"] + r["pred_q"] for r in sub])
devf = np.array([abs(r["beat"] - (r["d1"] * r["th"] + r["pred_q"])) for r in sub])
flf = np.array([r["fl_pred"] for r in sub])
lhs = np.array([r["beat"] / r["pred_q"] - 1.0 for r in sub])
rhs = np.array([2.0 * r["d1"] / (r["d2v"] * r["th"]) for r in sub])
print(f"  WITH the linear term restored: #(|beat - (d1*th + 0.5*k*th^2)| > floor) = "
      f"{int((devf > flf).sum())}/{len(sub)}   (without it, C1 was {len(c1)}/{len(lev)})")
print(f"     med |beat-(d1 th+pred_q)|/floor = {np.median(devf / flf):.3e}"
      f"   max = {(devf / flf).max():.3e}")
print(f"  the identity the two-term law predicts, ratio-1 = 2*d1/(k*th), per draw:")
print(f"     max |LHS - RHS| = {np.max(np.abs(lhs - rhs)):.3e}"
      f"   med |LHS| = {np.median(np.abs(lhs)):.3e}   med |RHS| = {np.median(np.abs(rhs)):.3e}"
      f"   Spearman(LHS, RHS) = {spear(lhs, rhs):+.3f}")
print(f"     -> the residual C1 measured is NOT a third-order term (pred3 explains {ncub}/{len(share)});")
print(f"        it is the first-order term d1*th, i.e. exactly the fact that v_b is not a critical point.")
loc = np.array([r["th"] / HREF for r in sub])
print(f"  LOCALITY of kappa: kappa is measured with stencil step h={HREF:.0e} at v_b, while the travel")
print(f"     to v* is theta.  #(theta > h) = {int((loc > 1).sum())}/{len(sub)},"
      f"  max theta/h = {loc.max():.2f},  med = {np.median(loc):.3f}")
print(f"     -> on those draws the law is an EXTRAPOLATION of a locally measured coefficient, not a")
print(f"        local statement, so a +-1% agreement there is stronger than registered, not weaker;")
print(f"        the matching test (h adapted so that h >= theta) is the next registration.")
worst = np.argsort(-np.abs(lhs))[:3]
print(f"     three largest |ratio-1|: " + "; ".join(
    f"#{worst[i]+1} theta/h={loc[worst[i]]:.2f} lin share={ls[worst[i]]:.2e}" for i in range(3)))



print("\n" + "=" * 132)
print("[P4] every draw (units: gap for beat/pred/floor; lambda for k, d1)")
print("=" * 132)
print(f"  {'#':>3} {'th':>10} {'beat/gap':>10} {'deficit':>8} {'|d1|':>10} {'lin share':>10} "
      f"{'k_values':>11} {'k_grad':>11} {'|k_v/k_g-1|':>11} {'pred_q/gap':>11} {'fl/gap':>10} "
      f"{'pred/fl':>9} {'ratio':>8} {'|dev|/fl':>9} {'d3':>11} {'pred3/dev':>10}")
for i, r in enumerate(rows, 1):
    ls = abs(r["d1"]) * r["th"] / r["beat"] if r["beat"] > 0 else np.nan
    dev = abs(r["beat"] - r["pred_q"]) / r["fl_pred"] if r["fl_pred"] > 0 else np.nan
    defc = (r["lam_b"] - r["true"]) / r["gap"]
    print(f"  {i:>3} {r['th']:>10.3e} {r['beat'] / r['gap']:>10.3e} {defc:>8.4f} {abs(r['d1']):>10.3e} "
          f"{ls:>10.3e} {r['d2v']:>11.4e} {r['d2g']:>11.4e} {abs(r['d2v'] / r['d2g'] - 1.0):>11.3e} "
          f"{r['pred_q'] / r['gap']:>11.3e} {r['fl_pred'] / r['gap']:>10.3e} "
          f"{r['pred_q'] / r['fl_pred']:>9.2e} {r['beat'] / r['pred_q']:>8.4f} {dev:>9.3e}")

print("\n" + "=" * 132)
print("[DIAGNOSTIC -- NOT REGISTERED, cannot be quoted as a claim]")
print("=" * 132)
th = np.array([r["th"] for r in rows]); bg = np.array([r["beat"] / r["gap"] for r in rows])
rt = np.array([r["beat"] / r["pred_q"] for r in rows if r["pred_q"] > 0 and np.isfinite(r["pred_q"])])
sl = float(np.polyfit(np.log(th) - np.log(th).mean(), np.log(bg) - np.log(bg).mean(), 1)[0]) \
    if len(rows) >= 3 and np.all(th > 0) and np.all(bg > 0) else float("nan")
print(f"  pooled log beat vs log th over ALL 40 draws = {sl:.3f} -- PRINTED ONLY TO BE EXPLICITLY NOT USED:")
print("     55.5 item 4 forbids a pooled slope as evidence for or against the law (kappa is itself")
print("     correlated with th within the family, so a slope mixes the exponent with the coefficient).")
print(f"  ratio spread over all draws with pred_q>0: med={np.median(rt):.3f} "
      f"min={rt.min():.4f} max={rt.max():.4f}  (n={rt.size})")
print(f"  theta here vs b63b DEGEN: this run med={np.median(th):.3e}; b63b printed median 4.912e-03 --")
print("     the streams differ (61206 here, 61204 there) so this is a CONSISTENCY reading, not a claim.")

print("\n" + "=" * 132)
print("判读（预登记，跑前写下；跑后只填数，不改方向）")
print("  P1 是第一结果：`kappa` 的两条路线（值 Richardson vs 梯度）若在本族上 disagreement 相对自身")
print("     超过 5%/25%，那 `beat/(½kappa th²)` 的分母就不是一个被测出的量，[B-CLAIM-65] 直接作废；")
print("     我不许自己在那之后还去报 ratio 的位数。")
print("  P2 的停止条件 (<20/40 有杠杆) 是 55.8 原样：触发就不是负结果，而是**本族上律未被检验**，")
print("     正文保持 b61 现在的措辞（律只在 10/50 上有杠杆），并把这句写进 open item 1。")
print("  P3 才是 claim：C1（|beat-pred_q|>floor）在 b61 的 50 抽样上是 0；本轮的近重合族里若有，")
print("     那就是二项展开在 `v*` 与 `v_b` 几乎同向、但 θ 比 b61 大两个数量级处失效 —— 这比 C3")
print("     出带更硬：C1 说展开不对，C3 只说常数不对。")
print("  C4 是混淆项计数：`|d1|th > 0.1 beat` 的抽样剔除并计数，绝不平均掉（b61 K3 的教训：")
print("     v_b 未达驻点时我拿一次项主导的数去验二次律，是量具在骗我）。")
print("  注意本轮把 55.8 的 ±2% 带和 floor 摆在一起看：pred 的地板必须 <1% 的 pred 才谈得上分辨 2%，")
print("     所以 band-resolvable(100x) 是**我为读带而加的第二计数**，跑前写在本文件 docstring 里，")
print("     它只会减少可报的抽样，不会增加（rule 17：能加通过数的改动才需要撤回）。")
print("  不注册：θ 的大小、deficit 的大小、pooled 斜率、组内 ρ_A（b63b 已证它在固定余弦下不可变）。")

print("\n" + "=" * 132)
print("IMPLEMENTATION / REGISTRATION HISTORY (post-run additions, disclosed per the stick-change rule)")
print("=" * 132)
print("  1. The FIRST run's G1 printed 1.978 and VOIDed the round.  Cause: I wrote the check as")
print("     |cos(th)v_b+sin(th)u - v*| with u from the SIGNED projection w = v* - (v_b.v*)v_b, but")
print("     lambda_1(v) depends on v only through B(v), C(v), so lambda_1(-v) = lambda_1(v) and the")
print(f"     descent returns either sign -- here {nneg}/{len(rows)} draws came back with v_b.v* < 0.")
print("     Fixed by pointing u at sign(v_b.v*)*v*.  THIS CHANGED A SENTENCE, NOT THE MEASUREMENT:")
print("     kappa's two routes are symmetric stencils in h, so u -> -u leaves them unchanged")
print(f"     ([S1] prints both differences as {dk.max():.3e} and {dg.max():.3e}), d1 flips sign exactly")
print(f"     (|d1(u_new)+d1(u_old)| = {dd1.max():.3e} on the flipped draws), and the registered ratio")
print(f"     uses only kappa and theta, so all 40 ratios agree with the old convention to "
      f"{rq.max():.3e}.")
print("     The claim's numbers are the ones the first run would have produced had I written the")
print("     check correctly.")
print("  2. The [P3b] residual block (third derivative by a 4-point antisymmetric stencil, its")
print("     Richardson floor, the cubic share, and the Spearman of |ratio-1| against its candidate")
print("     regressors) was added AFTER P1-P3 printed.  It consumes NO rng draws (it only calls")
print("     lam_of_v on designs already built), it registers nothing, and rule 19 is applied inside it:")
print("     every regressor's range is printed before any correlation is read, because in b63b I")
print("     registered a Spearman against a quantity that could not vary.")
print("  3. NOT a post-hoc change: the band-resolvable gate (pred_q > 100*floor) was written into the")
print("     docstring BEFORE the first run, with the reason (a +-2% band cannot be read with a floor at")
print("     10% of the prediction).  It can only REMOVE draws from the band count, never add them")
print("     (rule 17: a gate that adds passes is the one that needs a retraction).  Here it removes none.")
print("  4. The testable set, the 10x leverage gate, the 6-start set, the exclusion of linear-share > 0.1")
print("     and the [0.98, 1.02] band are exactly as registered in community.md 55.8; none of them was")
print("     touched after the numbers existed.")
print("  5. Prose in this file hard-coded three counts (C1's, the cubic's, and kappa's span) instead of")
print(f"     binding them: they read {len(c1)}/{len(lev)}, {ncub}/{len(share)}, "
      f"{max(r['d2v'] for r in sub) / min(r['d2v'] for r in sub):.1f}x.  The VALUES were right, the")
print("     PRACTICE was not -- an unbound literal in a printout is a claim that cannot survive a")
print("     re-run, and community.md 56.4 item 4 records what that style cost me this round.")

print("\n" + "=" * 132)
print("判读（跑后填数，方向为上面预登记的原话；只填数，不改方向）")
print("=" * 132)
rt_lev = np.array([r["beat"] / r["pred_q"] for r in lev])
print(f"  P1: 未触发作废 —— |k_v/k_g-1| med={np.median([abs(r['d2v'] / r['d2g'] - 1.0) for r in lev]):.3e}"
      f" <= 0.05, max={max(abs(r['d2v'] / r['d2g'] - 1.0) for r in lev):.3e} <= 0.25。所以 ratio 的分母")
print("      是一个被测出的量，P3 的位数才允许引用。")
print(f"  P2: 停止条件未触发（有杠杆 {len(lev)}/{len(tex)}，注册门槛 20）。=> 与 b61 的 10/50 不同，")
print("      律在本族**被检验了**，所以 b61 的措辞可以被本轮替换（这是本轮的实质收获）。")
print(f"  P3: C3 未触发（ratio med={np.median(rt_lev):.4f} 在带内），[B-CLAIM-65] **按注册成立**；")
print(f"      但 C1 触发 {len(c1)}/{len(lev)}，且 C1 用的是我自己两路线的地板，所以这个偏差是真的：")
print("      **律是 ±1% 的律，不是恒等式**。我不用 C1 去否证自己注册的带（两者问的是不同问题：")
print("      C3 问常数，C1 问展开是否到此为止），也不把 C1 藏起来。")
print(f"  C4: 剔除 {nex - len(tex)} 个（一次项主导），所以本轮的 40 个可测抽样里没有'拿一次项主导的数")
print("      去验二次律'这一类污染 —— 然而 [P3b] 显示一次项仍以 |d1|th/beat ~ 1e-4 的量级在场，")
print(f"      恢复它把 C1 从 {len(c1)}/{len(lev)} 降到 {int((devf > flf).sum())}/{len(sub)}。这两件事不矛盾：")
print("      门槛 0.1 是**混淆项**的门槛，地板 1 是**测量**的门槛，后者灵敏三个数量级。")
print(f"  预登记里的三条方向，逐条对数：C1=0 的预期（b61 的 50 抽样）在本族不成立（{len(c1)}/{len(lev)}）；")
print(f"      三次项候选被否（解释 {ncub}/{len(share)}）；一次项候选被支持（身份 ratio-1=2d1/(k th) 的")
print(f"      Spearman={spear(lhs, rhs):+.3f}，且 |ratio-1| 对 linear share 的 Spearman="
      f"{spear(rr, np.array([abs(r['d1']) * r['th'] / r['beat'] for r in sub])):+.3f}）。")
print(f"  下一步（不注册在本轮）：{int((loc > 1).sum())}/{len(sub)} 个抽样的 theta 超过模板步长 h，")
print("      所以 kappa 在那里是外推。把 h 自适应到 h>=theta 的检验已注册为 community.md 56.9 的 [B-CLAIM-67]。")
