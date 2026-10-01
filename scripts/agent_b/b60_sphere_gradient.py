"""b60 -- the outer leg: manifold gradient over the sphere of directions, inner value exact.

Why this exists (the sentence I had to take back):  b59's (vii) said rays through v1 sweep only a
"two-plane family".  Wrong as stated -- every unit v IS cos(theta)v1 + sin(theta)u with u perp v1, so
the UNION over u of those rays is the whole sphere, and b59 sampled three u per draw.  The inner
problem is exact (G, the cleared stationary numerator, verified exact-against-exact at 9.6e-16 in b59);
what is missing is a real search over u.  This script is that search.

OWN RNG STREAM: seed 61200, rebound in b57's namespace BEFORE any draw.  Therefore no per-draw
comparison with b57/b59/b53's shared-draw tables is implied -- only within-draw comparisons made here.

GATES (any failure voids everything below it -- retraction (p)'s failure mode made structural):
  G1 evaluator: lam_-(B,C) by the rationalised root must match np.roots on the SAME quadratic (a
      companion-matrix route, independent of my algebra) to <=1e-13 in lambda on 200 visited v.
      >1e-13 => my evaluator is wrong and nothing below means anything.
  G2 anchor:   |lam_1(v1) - cf_1| <= 1e-15 on every draw.
  G3 gradient, EXACT vs EXACT (rule 16: never a finite difference as referee for an identity):
      on the steepest ray, grad_sphere(v_theta) . d v_theta/d theta must equal
      (1+x^2) * dlam_exact(x)  (b59's closed-form derivative of the rationalised root) to <=1e-8
      relative with scale |lam|/x.  A failure means 2*lam_B*A0 v + 2*lam_C*B2 v is not grad lam_-.

FALSIFIERS (directions registered before the numbers exist):
  F1  min over the sphere BELOW lam_1(v*):  then v* is NOT the conjectured infimiser and open item 1
      moves.  Report the margin in units of the gap AND in lambda.
  F2  min over the sphere BELOW lambda_min(G) minus the draw's own 100*eps*||G||_2 floor:  a math
      contradiction with eq:chain1, not a numerics complaint.  Must be 0 draws.
  F3  the converged value depends on the START (v1 / v* / top eigenvector of B2 / random) by more than
      1e-6 of the gap:  then lam_1 is not unimodal on the sphere and every "best direction" statement
      must be relabelled "a local minimum found from this start".
  F4  a 200,000-point random-direction grid beating the optimiser by more than 10x the measured
      two-route evaluator floor:  my optimiser stopped at a local min and the search is void.
  F5  ||grad_sphere|| at the reported minimum above 1e-6*|lam|:  not converged.
  NESTED (not a falsifier of the search but of the bookkeeping): the sphere minimum cannot EXCEED
      b59's exact 3-ray minimum on the same draw, because more directions cannot raise an infimum.
  NOT registered (no prediction, only measurement): the deficit of the sphere minimum over
      lambda_min(G).  b59's sampled planes gave 0.2357 of the gap as the smallest deficit; more u does
      not mean a smaller infimum, so this can come out anywhere.
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")     # the 判读 block is Chinese; do not depend on the console cp

_ns = {}
_src = open("b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)
_ns["rng"] = np.random.default_rng(61200)          # own stream, rebound before the first draw

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
setup = _ns["setup"]
ray_L = _ns["ray_L"]
scalars = _ns["scalars"]
stats = _ns["stats"]
crossings = _ns["crossings"]
SPECTRA = _ns["SPECTRA"]
rng = _ns["rng"]

_bns = {}                                          # b59's head, in its own namespace
_bns["__file__"] = "b59_setlevel_gridfree.py"      # its head resolves b57's path through __file__
_b59 = open("b59_setlevel_gridfree.py", encoding="utf-8").read()
exec(compile(_b59.split("# ---------------------------------------------------------------- cell driver")[0],
             "b59-head", "exec"), _bns)
crossings_stable = _bns["crossings_stable"]
dlam_exact = _bns["dlam_exact"]
ray_min_exact = _bns["ray_min_exact"]


def lam_of_v(s, v):
    """lambda_1(v) = lam_-(B(v), C(v)), rationalised route (b59's evaluator)."""
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    return float(crossings_stable(np.array([B]), np.array([C]))[0]), B, C


def lam_roots(B, C):
    """INDEPENDENT route: companion-matrix roots of C*lam^2-(B+C)*lam+B(1-B)=0."""
    r = np.roots([C, -(B + C), B * (1.0 - B)])
    ok = r[np.abs(r.imag) < 1e-9].real
    ok = ok[(ok > 0.0) & (ok < 1.0)]
    return float(np.min(ok)) if ok.size else np.nan


def grad_sphere(s, v):
    """grad of lam_-(B(v),C(v)) on the sphere, by implicit differentiation of F(lam,B,C)=0:
    F_lam = 2C lam-(B+C) = -sqrt(disc) on the lam_- branch, F_B = 1-lam-2B, F_C = lam^2-lam."""
    lam, B, C = lam_of_v(s, v)
    disc = (B + C) ** 2 - 4.0 * C * B * (1.0 - B)
    Fl = 2.0 * C * lam - B - C                      # = -sqrt(disc)
    if abs(Fl) < 1e-14:
        return lam, np.zeros_like(v), B, C, np.sqrt(max(disc, 0.0))
    lam_B = -(1.0 - lam - 2.0 * B) / Fl
    lam_C = -(lam * lam - lam) / Fl
    g = 2.0 * lam_B * (s["A0"] @ v) + 2.0 * lam_C * (s["B2"] @ v)
    return lam, g - v * float(v @ g), B, C, np.sqrt(max(disc, 0.0))


def sphere_descent(s, v0, iters=800):
    """backtracking retraction v <- normalise(v - t*grad), accepting only strict decreases of lam.
    Stops when the accepted step changes lambda by <1e-17*|lam| (the evaluator cannot see further),
    so a stalled start is reported by its ||grad||, not assumed to be a minimum."""
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
                    return float(lam), v, float(np.linalg.norm(grad_sphere(s, v)[1])), True
                break
        if not improved:
            break
    lam, gs = grad_sphere(s, v)[:2]
    return float(lam), v, float(np.linalg.norm(gs)), False


def perp_random(s):
    w = rng.normal(size=s["A0"].shape[0])
    w = w - s["v1"] * float(s["v1"] @ w)
    n = float(np.linalg.norm(w))
    return w / n if n > 0.0 else None


CELLS = [
    ("q=3 spread 5-75", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 5-75"])))(lam_spd(6))),
    ("q=3 spread 2-45-88", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))(lam_spd(6))),
    ("q=3 spread 30-88-89", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 30-88-89"])))(lam_spd(6))),
    ("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)),
                                                          np.cos(np.deg2rad(40))])))(lam_spd(5))),
    ("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(t))
                                                           for t in (5, 25, 55, 80)])))(lam_spd(7))),
]

NGRID = 200000
g1_err, g2_err, g3_err = [], [], []
rows = {tag: [] for tag, _ in CELLS}
f1_beats_vstar = f2_below_floor = f4_grid_beats = 0
vstar_margin, grid_margin, deficit_all, spread_all, nested_all = [], [], [], [], []

print("=" * 132)
print("[S0] gates.  G1 rationalised root vs np.roots (independent evaluator);")
print("     G2 |lam_1(v1)-cf_1|;  G3 EXACT-vs-EXACT gradient: grad_sphere.v' == (1+x^2) dlam_exact")
print("=" * 132)

for tag, gen in CELLS:
    for _d in range(10):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        gap = s["cf1"] - s["true"]
        # ---- G2
        g2_err.append(abs(lam_of_v(s, s["v1"])[0] - s["cf1"]))
        # ---- G1 on 200 visited points (v1, v*, 198 random)
        pts = [s["v1"], s["vstar"]] + [rng.normal(size=s["A0"].shape[0]) for _ in range(198)]
        for w in pts:
            w = w / np.linalg.norm(w)
            lam, B, C = lam_of_v(s, w)
            alt = lam_roots(B, C)
            if np.isfinite(alt):
                g1_err.append(abs(lam - alt))
        # ---- G3 on the steepest ray, exact against b59's closed-form derivative
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
                    g3_err.append(abs(float(gs @ dv) - (1.0 + x * x) * dl) / (abs(lam0) / max(x, 1e-12)))
        # ---- the search: starts v1, v*, top eigenvector of B2, three random
        starts = {"v1": s["v1"], "vstar": s["vstar"], "eigB2": np.linalg.eigh(s["B2"])[1][:, -1],
                  "r1": rng.normal(size=s["A0"].shape[0]),
                  "r2": rng.normal(size=s["A0"].shape[0]),
                  "r3": rng.normal(size=s["A0"].shape[0])}
        res = {k: sphere_descent(s, w0) for k, w0 in starts.items()}
        best_key = min(res, key=lambda k: res[k][0])
        lam_best, v_best, gn_best, _tiny = res[best_key]
        # F3 is about MULTIPLICITY, not about stalled starts: only starts whose final ||grad|| is
        # small relative to |lam| may be compared, and the statistic is their spread.
        conv = [k for k in res if res[k][2] <= 1e-4 * abs(res[k][0])]
        spread_conv = (max(res[k][0] for k in conv) - min(res[k][0] for k in conv)) \
            if len(conv) >= 2 else np.nan
        # ---- F4: a random grid can only OVERESTIMATE the true infimum
        Vg = rng.normal(size=(NGRID, s["A0"].shape[0]))
        Vg /= np.linalg.norm(Vg, axis=1, keepdims=True)
        Bg = np.einsum("ij,jk,ik->i", Vg, s["A0"], Vg)
        Cg = np.einsum("ij,jk,ik->i", Vg, s["B2"], Vg)
        lg = crossings_stable(Bg, Cg)
        grid_min = float(np.nanmin(lg))
        fl = float(np.nanmax(np.abs(lg - crossings(Bg, Cg))))    # measured evaluator noise on this draw
        # ---- nested: b59's exact inner solver on 3 rays of the SAME draw
        rays = ([u for u in [uu[0]] if uu is not None] +
                [x for x in (perp_random(s), perp_random(s)) if x is not None])
        ray_best = min([m for m in (ray_min_exact(s, scalars(s, u))[0] for u in rays)
                        if np.isfinite(m)] or [np.inf])
        lam_vstar = lam_of_v(s, s["vstar"])[0]
        rows[tag].append(dict(gap=gap, lam_best=lam_best, spread=spread_conv, nconv=len(conv),
                              gn=gn_best,
                              grid_min=grid_min, fl=fl, ray_best=ray_best, lam_vstar=lam_vstar,
                              true=s["true"], cf1=s["cf1"], ang_vstar=float(
                                  np.arccos(min(1.0, abs(v_best @ s["vstar"])))),
                              deficit=(lam_best - s["true"]) / gap, key=best_key,
                              beat=(lam_vstar - lam_best) / gap,
                              beat_fl=(lam_vstar - lam_best) / fl if fl > 0 else np.nan,
                              ang_best=float(np.arccos(min(1.0, abs(v_best @ s["vstar"]))))))
        if lam_best < lam_vstar:
            f1_beats_vstar += 1
            vstar_margin.append((lam_vstar - lam_best) / gap)
        if lam_best < s["true"] - s["floor"]:
            f2_below_floor += 1
        if grid_min < lam_best - 10.0 * fl:
            f4_grid_beats += 1
            grid_margin.append((lam_best - grid_min) / gap)
        deficit_all.append((lam_best - s["true"]) / gap)
        spread_all.append(spread_conv / gap)
        nested_all.append((lam_best - ray_best) / gap)

print(f"  G1 max |rationalised - np.roots| over visited v   = {max(g1_err):.3e}   (gate <=1e-13)"
      f"   n={len(g1_err)}   -> {'PASS' if max(g1_err) <= 1e-13 else 'VOID: evaluator wrong'}")
print(f"  G2 max |lam_1(v1) - cf_1|                         = {max(g2_err):.3e}   (gate <=1e-15)"
      f"   -> {'PASS' if max(g2_err) <= 1e-15 else 'VOID'}")
print(f"  G3 max |grad.v' - (1+x^2)dlam_exact| / (|lam|/x)  = {max(g3_err):.3e}   (gate <=1e-8)"
      f"   n={len(g3_err)}   -> "
      f"{'PASS: 2*lam_B*A0v + 2*lam_C*B2v IS the sphere gradient' if max(g3_err) <= 1e-8 else 'VOID: the search below is meaningless'}")

print("\n" + "=" * 132)
print("[S1] the search, PER DRAW within cell (rule 13: no pooling across cells).")
print("     deficit = (sphere min - lambda_min(G))/gap;  0 would mean the infimum ATTAINS lambda_min(G).")
print("     beat v* = draws whose sphere min is strictly below lambda_1(v*);  F4 = 200k grid undercut;")
print("     spread = max minus min over the CONVERGED starts only (a stalled start is not evidence")
print("     of a second minimum);  nested = sphere min minus b59's 3-ray min")
print("              (must be <=0: adding directions cannot raise an infimum).")
print("=" * 132)
print(f"  {'cell':>22} {'n':>3} {'min/med/max deficit/gap':>26} {'beat v*':>8} {'F4':>3} "
      f"{'spread/gap':>13} {'max|grad|':>10} {'nested max':>11} {'argmin(v*,v1)':>13} {'#conv':>8}")
for tag, _ in CELLS:
    a = rows[tag]
    if not a:
        print(f"  {tag:>22}   0  (all draws rejected by setup)")
        continue
    df = np.array([r["deficit"] for r in a])
    sp = np.array([r["spread"] / r["gap"] for r in a], dtype=float)
    spm = float(np.nanmax(sp)) if bool(np.isfinite(sp).any()) else float("nan")
    bv = sum(1 for r in a if r["lam_best"] < r["lam_vstar"])
    f4 = sum(1 for r in a if r["grid_min"] < r["lam_best"] - 10.0 * r["fl"])
    gn = max(r["gn"] for r in a)
    nm = max((r["lam_best"] - r["ray_best"]) / r["gap"] for r in a)
    nk = sum(1 for r in a if r["key"] in ("v1", "vstar"))
    nc = sum(r["nconv"] for r in a)
    print(f"  {tag:>22} {len(a):>3} {df.min():.4f} /{np.median(df):.4f} /{df.max():.4f}"
          f" {bv:>8} {f4:>3} {spm:>13.3e} {gn:>10.2e} {nm:>11.3e} {nk:>6}/{len(a)}"
          f" {nc:>3}/{6*len(a)}")

print("\n" + "=" * 132)
print("[S2] the falsifiers, counted over all draws:")
print(f"  F1 #(sphere min < lambda_1(v*))         = {f1_beats_vstar}   "
      f"largest beat = {(max(vstar_margin) if vstar_margin else 0.0):.6e} of gap")
print(f"  F2 #(sphere min < lambda_min(G)-floor)  = {f2_below_floor}   (MUST be 0: eq:chain1)")
_sc = np.array(spread_all, dtype=float)
_fin = _sc[np.isfinite(_sc)]
print(f"  F3 spread over CONVERGED starts (|grad|<=1e-4|lam|), units of gap: max={(_fin.max() if _fin.size else float('nan')):.3e}, "
      f"#(>1e-6) = {int(np.sum(_fin > 1e-6))} of {int(np.isfinite(_sc).sum())} testable draws "
      f"({int(np.sum(~np.isfinite(_sc)))} had <2 converged starts -- not testable, reported not guessed)   -> "
      f"{'starts land on DIFFERENT local minima' if _fin.size and _fin.max() > 1e-6 else 'converged starts agree'}")
print(f"  F4 #(grid undercuts optimiser by >10x evaluator floor) = {f4_grid_beats}   "
      f"largest = {(max(grid_margin) if grid_margin else 0.0):.6e} of gap")
allr = [x for a in rows.values() for x in a]
print(f"  F5 #(reported min with |grad_sphere| > 1e-6*|lam|) = "
      f"{sum(1 for r in allr if r['gn'] > 1e-6 * abs(r['lam_best']))}")
_nested = [(r["lam_best"] - r["ray_best"]) / r["fl"] for r in allr]
print(f"  nested max (sphere min minus 3-ray min) = {max(nested_all):.3e} of gap; "
      f"in the DRAW's own evaluator floor: max = {max(_nested):.3e}, #(>10 floors) = "
      f"{int(np.sum(np.array(_nested) > 10.0))}   (a positive excess beyond that would mean my "
      f"optimiser is worse than b59's exact inner solver)")
print(f"  angle(v_best, v*) : med={np.median([r['ang_vstar'] for r in allr]):.6f} rad "
      f"max={max(r['ang_vstar'] for r in allr):.6f} rad")
print(f"  [not registered] deficit of sphere min over lambda_min(G): {stats(np.array(deficit_all))}")

print("\n" + "=" * 132)
print("[S3] F1 in detail -- the beat over lambda_1(v*) is attained by a CONCRETE direction, so it does")
print("     not depend on my optimiser being globally converged; but it must clear the evaluator floor.")
print("     beat/gap and beat/floor are per draw; 'ang' is the angle between that direction and v*.")
print("=" * 132)
print(f"  {'cell':>22} {'#beat':>6} {'max beat/gap':>13} {'max beat/floor':>15} "
      f"{'ang at max beat':>16} {'med ang':>10}")
tot_fl = []
for tag, _ in CELLS:
    a = [r for r in rows[tag] if r["beat"] > 0.0]
    if not a:
        print(f"  {tag:>22} {0:>6}   (no draw beaten)")
        continue
    tot_fl += [r["beat_fl"] for r in a]
    k = max(a, key=lambda r: r["beat"])
    print(f"  {tag:>22} {len(a):>6} {k['beat']:>13.6e} {max(r['beat_fl'] for r in a):>15.3e} "
          f"{k['ang_best']:>16.3e} {float(np.median([r['ang_best'] for r in a])):>10.3e}")
_tf = np.array(tot_fl, dtype=float)
print(f"  overall: #(beat > 10x that draw's evaluator floor) = {int(np.sum(_tf > 10.0))}/{_tf.size}"
      f"   min beat/floor = {np.nanmin(_tf):.3e}")

print("\n" + "=" * 132)
print("判读（预登记，跑前写下的方向；跑后只填数，不改方向）")
print("  G1/G3 都是**两条独立解析路线互验**（companion 矩阵的根、b59 对理化根的闭式导数），")
print("     有限差分不参与门禁 —— 规则 16。G3 FAIL 意味着 lam_B/lam_C 或 F_lam=-sqrt(disc) 用错，")
print("     那么下面的搜索全是无意义的，我会直接宣布本轮作废。")
print("  F1 是本轮唯一能推动 open item 1 的结果：>0 且幅度显著 => v* 不是下确界方向。")
print("     F1=0 => 我拿到的是「v* 在一次逐维收敛的独立优化下仍未被击败」，正文按这句写。")
print("  F2 是数学判据（eq:chain1 对每个单位 v 成立），不是数值判据；出现 1 条就要查评测器。")
print("  F4 与 nested 是**两个方向**的钳制：随机网只能高估下确界（网格更低=优化器卡局部），")
print("     b59 的 3 射线内层只能高估球面下确界（球面更大=我的优化器不如精确内层，本轮不可用）。")
print("  F3 触发则「最好方向」必须改成「从该起点找到的局部极小」，并计入撤回清单。")
