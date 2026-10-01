"""b62 -- [B-CLAIM-63]: the angle by which v* misses the lambda_1-infimiser is ONE NEWTON STEP.

From b61: beat = lambda_1(v*) - min_v lambda_1(v) = 0.5*kappa*th^2 with th = angle(v*, v_b), measured
per draw with real leverage on only 10/50 draws (the other 40 have 0.5*kappa*th^2 below the floor of
its own estimate -- the floor carries a factor th).  This script moves the whole measurement to v*,
where the relevant coefficient has a floor that does NOT depend on th:

    g*  = || grad_S lambda_1 (v*) ||       (nonzero exactly BECAUSE v* is not the infimiser)
    k*  = curvature of lambda_1 on the sphere at v* along the descent direction u = -grad/g*
    along that geodesic:  lambda(h) = lambda(0) - g* h + 0.5 k* h^2 + (1/6) m* h^3 + ...
    stationarity:  -g* + k* h + 0.5 m* h^2 = 0
      =>  h_newton = g*/k*                          (claim, pure)
      =>  h_exact  = 2 g* / (k* + sqrt(k*^2 - 2 m* g*))    (cubic-corrected, small |m*|)

So the claim is: th_measured = angle(v*, v_b) equals g*/k* to within the measured floors, and
beat = lambda_1(v*) - lambda_1(v_b) equals g*^2/(2k*).  Both coefficients are estimated at v* by TWO
independent routes (analytic gradient vs Richardson value differences; value curvature vs curvature
from gradient differences), so each has a measured floor.

FALSIFIERS (registered before the numbers exist):
  N1  #(|th_meas - g*/k*| > 10*floor): pure NEWTON FAILS.  If the cubic-corrected h_exact passes where
      the pure one fails, the claim survives only in the corrected form -- and must be written that way.
  N2  #(k* <= 0 at v*): no descent direction of the Newton form; those draws are excluded, reported.
  N3  #(|beat - g*^2/(2k*)| > 10*floor): the priced beat fails -- a direct contradiction of b61's law
      at the same draws, which would void both.
  N4  #(the descent from v* lands on a DIFFERENT minimum than b60's 6-start best by >10 floors):
      multiplicity, i.e. b60's single-basin statement does not extend to a directed step.
  N5  the point of the round: the LEVERAGE COUNT.  If fewer than 20/50 draws have g*/k* resolved above
      10*floor, the whole Newton formulation buys nothing over b61 and the item is dropped (53.10).
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ns = {}
_src = open("b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)
_ns["rng"] = np.random.default_rng(61202)          # own stream (b60 61200, b61 61201)

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
setup = _ns["setup"]
stats = _ns["stats"]
crossings = _ns["crossings"]
SPECTRA = _ns["SPECTRA"]
rng = _ns["rng"]

_b1 = {}
_b1["__file__"] = "b61_geodesic_law.py"
_b1src = open("b61_geodesic_law.py", encoding="utf-8").read()
exec(compile(_b1src.split("CELLS = [")[0], "b61-head", "exec"), _b1)
# b61 defines HREF AFTER its CELLS block, so the head split cuts it; d1_grad and d2_value look it up
# in their own globals at call time, so it must be injected into _b1, not just into this file.
_b1["HREF"] = 1e-2
lam_of_v = _b1["lam_of_v"]
grad_sphere = _b1["grad_sphere"]
sphere_descent = _b1["sphere_descent"]
geo = _b1["geo"]
lam_roots = _b1["lam_roots"]
crossings_stable = _b1["crossings_stable"]
d1_grad = _b1["d1_grad"]
d2_value = _b1["d2_value"]
d2_grad = _b1["d2_grad"]
HREF = _b1["HREF"]

CELLS = [
    ("q=3 spread 5-75", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 5-75"])))(lam_spd(6))),
    ("q=3 spread 2-45-88", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))(lam_spd(6))),
    ("q=3 spread 30-88-89", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 30-88-89"])))(lam_spd(6))),
    ("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)),
                                                          np.cos(np.deg2rad(40))])))(lam_spd(5))),
    ("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(t))
                                                           for t in (5, 25, 55, 80)])))(lam_spd(7))),
]


def coeff_at_vstar(s, v, u):
    """g*, k*, m* along the geodesic v(h)=cos h v + sin h u, each by two independent routes."""
    lam0 = lam_of_v(s, v)[0]
    ga = float(grad_sphere(s, v)[1] @ u)                    # analytic (b60/b61's verified gradient)
    def val(hw):
        return lam_of_v(s, geo(v, u, hw))[0]
    h, hh = HREF, HREF / 2.0
    d1a = (val(h) - val(-h)) / (2 * h)                      # value routes
    d1b = (val(hh) - val(-hh)) / (2 * hh)
    g_v = (4 * d1b - d1a) / 3.0
    fl_g = max(abs(ga - g_v), abs(d1a - d1b) / 3.0)
    k = lambda hw: (val(hw) - 2 * lam0 + val(-hw)) / (hw * hw)
    k1, k2 = k(h), k(hh)
    k_v = (4 * k2 - k1) / 3.0
    a = float(grad_sphere(s, geo(v, u, h))[1] @ u)
    b = float(grad_sphere(s, geo(v, u, -h))[1] @ u)
    k_g = (a - b) / (2 * h)
    fl_k = max(abs(k_v - k_g), abs(k1 - k2) / 3.0)
    m_v = (val(2 * h) - 2 * val(h) + 2 * val(-h) - val(-2 * h)) / (2 * h ** 3)
    m_a = (a - b) / (2 * h)                                 # = k_g, cubic NOT separable this way
    fl_m = float("nan")
    return dict(lam0=lam0, ga=ga, g_v=g_v, fl_g=fl_g, k_v=k_v, k_g=k_g, fl_k=fl_k,
                m_v=m_v, fl_m=fl_m, B=float(v @ s["A0"] @ v), C=float(v @ s["B2"] @ v))


rows = {tag: [] for tag, _ in CELLS}
for tag, gen in CELLS:
    for _d in range(10):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        gap = s["cf1"] - s["true"]
        vs = s["vstar"]
        lam, gs = grad_sphere(s, vs)[:2]
        nrm = float(np.linalg.norm(gs))
        u = -gs / nrm if nrm > 0.0 else None               # descent direction on the sphere
        co = coeff_at_vstar(s, vs, u) if u is not None else None
        # the measured answer: where does the directed step actually land, and where does the
        # 6-start search land (b60's best)?
        th_meas = beat_meas = np.nan
        th_dir = lam_dir = np.nan
        if co is not None:
            res = [sphere_descent(s, w0) for w0 in
                   (s["v1"], vs, np.linalg.eigh(s["B2"])[1][:, -1],
                    rng.normal(size=s["A0"].shape[0]), rng.normal(size=s["A0"].shape[0]),
                    rng.normal(size=s["A0"].shape[0]))]
            best = min(res, key=lambda r: r[0])
            lam_b, v_b = best[0], best[1]
            # N4: does the descent started AT v* land on the same minimum the 6-start search finds?
            # Compared in this draw's own evaluator floor (rationalised root vs np.roots at both ends).
            res_vs = res[1]
            a_b, B_b, C_b = lam_of_v(s, v_b)
            a_s, B_s, C_s = lam_of_v(s, res_vs[1])
            fl_lam = max(abs(a_b - lam_roots(B_b, C_b)), abs(a_s - lam_roots(B_s, C_s)))
            th_meas = float(np.arccos(min(1.0, abs(float(vs @ v_b)))))
            beat_meas = lam - lam_b
            hn = abs(co["ga"]) / co["k_v"] if co["k_v"] > 0 else np.nan
            disc = co["k_v"] ** 2 - 2.0 * co["m_v"] * abs(co["ga"])
            hx = 2.0 * abs(co["ga"]) / (co["k_v"] + np.sqrt(disc)) if (disc > 0 and co["k_v"] > 0) else np.nan
            th_dir = float(hn)
            lam_dir = float(hx)
            fl_th = co["fl_k"] * hn / max(co["k_v"], 1e-300) + co["fl_g"] / max(co["k_v"], 1e-300) \
                if np.isfinite(hn) else np.nan
            beat_pred = co["ga"] ** 2 / (2.0 * co["k_v"]) if co["k_v"] > 0 else np.nan
            fl_beat = abs(co["ga"]) * co["fl_g"] / co["k_v"] + co["ga"] ** 2 * co["fl_k"] / \
                (2.0 * co["k_v"] ** 2) if np.isfinite(beat_pred) else np.nan
            rows[tag].append(dict(gap=gap, g=co["ga"], gv=co["g_v"], fl_g=co["fl_g"],
                                  k=co["k_v"], kg=co["k_g"], fl_k=co["fl_k"], m=co["m_v"],
                                  th=th_meas, beat=beat_meas, hn=hn, hx=hx, fl_th=fl_th,
                                  bp=beat_pred, fl_bp=fl_beat, lamstar=lam, lam_b=lam_b,
                                  nrm=nrm, d_vs=res_vs[0] - lam_b, fl_lam=fl_lam, key="d6"))
        else:
            rows[tag].append(dict(gap=gap, g=0.0, gv=0.0, fl_g=np.nan, k=np.nan, kg=np.nan,
                                  fl_k=np.nan, m=np.nan, th=0.0, beat=0.0, hn=np.nan, hx=np.nan,
                                  fl_th=np.nan, bp=np.nan, fl_bp=np.nan, lamstar=lam,
                                  lam_b=lam, nrm=nrm, d_vs=0.0, fl_lam=np.nan, key="flat"))

allr = [x for a in rows.values() for x in a]
print("=" * 132)
print("[S0] sanity: the analytic gradient at v* against the Richardson VALUE route, both curvature")
print("     routes at v*, and the evaluator at v* by np.roots")
print("=" * 132)
fg = np.array([r["g"] - r["gv"] for r in allr], dtype=float)
print(f"  max |g*_analytic - g*_value| = {np.nanmax(np.abs(fg)):.3e}   "
      f"max fl_g = {max(r['fl_g'] for r in allr):.3e}")
kr = np.array([r["k"] / r["kg"] for r in allr if r["kg"] != 0.0], dtype=float)
print(f"  k*_value / k*_gradient : med={np.nanmedian(kr):.6f} spread=[{np.nanmin(kr):.6f},{np.nanmax(kr):.6f}]")
print(f"  #(||grad_S lam_1(v*)|| == 0 to machine) = {sum(1 for r in allr if r['nrm'] == 0.0)}   "
      f"|grad| med={np.median([r['nrm'] for r in allr]):.3e} min={min(r['nrm'] for r in allr):.3e}")

print("\n" + "=" * 132)
print("[S1] per cell.  th_meas = angle(v*, v_b) from the 6-start search; g*/k* = the Newton step;")
print("     h_exact = cubic-corrected root.  Units: g*,k*,m* in lambda; th in rad; beat in units of gap.")
print("     IMPLEMENTATION HISTORY (disclosed, per rule 16 -- the registered gates are unchanged):")
print("     (1) first run used FIVE starts while N4 was registered against 'b60's 6-start best'; the")
print("         sixth random start is now added so the code matches the registration, and N4 (which the")
print("         first run never computed) is now printed;")
print("     (2) b61 defines HREF after its CELLS block, so this file's head-split of it cut that")
print("         constant; it is re-injected into b61's namespace (same value 1e-2) -- the first run died")
print("         on this, so there is no earlier result affected;")
print("     (3) [S3] below is a print-only diagnostic added after the run, no gate or floor changed.")
print("=" * 132)
print(f"  {'cell':>22} {'n':>3} {'|g*| med':>10} {'k* med':>9} {'th_meas med':>12} {'g*/k* med':>10} "
      f"{'th/(g*/k*) med':>15} {'N1':>3} {'N2':>3} {'N3':>3} {'leverage':>9}")
for tag, _ in CELLS:
    a = [r for r in rows[tag] if r["key"] != "flat"]
    if not a:
        print(f"  {tag:>22} {0:>3}   (no draw with a descent direction)")
        continue
    rr = np.array([r["th"] / r["hn"] for r in a if np.isfinite(r["hn"]) and r["hn"] > 0], dtype=float)
    n1 = sum(1 for r in a if np.isfinite(r["fl_th"]) and abs(r["th"] - r["hn"]) > 10.0 * r["fl_th"])
    n2 = sum(1 for r in a if not (r["k"] > 0))
    n3 = sum(1 for r in a if np.isfinite(r["fl_bp"]) and r["fl_bp"] > 0
             and abs(r["beat"] - r["bp"]) > 10.0 * r["fl_bp"])
    lev = sum(1 for r in a if np.isfinite(r["hn"]) and r["hn"] > 10.0 * r["fl_th"])
    print(f"  {tag:>22} {len(a):>3} {np.median([abs(r['g']) for r in a]):>10.3e} "
          f"{np.median([r['k'] for r in a]):>9.3e} {np.median([r['th'] for r in a]):>12.3e} "
          f"{np.median([r['hn'] for r in a]):>10.3e} {np.nanmedian(rr) if rr.size else float('nan'):>15.4f} "
          f"{n1:>3} {n2:>3} {n3:>3} {lev:>3}/{len(a)}")

print("\n" + "=" * 132)
print("[S2] the falsifiers, over all draws")
print("=" * 132)
good = [r for r in allr if r["key"] != "flat"]
n1 = [r for r in good if np.isfinite(r["fl_th"]) and abs(r["th"] - r["hn"]) > 10.0 * r["fl_th"]]
print(f"  N1 #(pure Newton th != g*/k* beyond 10*floor) = {len(n1)}/{len(good)}"
      + (f"   worst |th - g*/k*|/floor = {max(abs(r['th'] - r['hn']) / r['fl_th'] for r in n1):.3e}" if n1 else ""))
n1c = [r for r in good if np.isfinite(r["hx"]) and np.isfinite(r["fl_th"])
       and abs(r["th"] - r["hx"]) > 10.0 * r["fl_th"]]
print(f"     cubic-corrected step h_exact: #(fails beyond 10*floor) = {len(n1c)}/{len(good)}")
print(f"  N2 #(k* <= 0 at v*) = {sum(1 for r in good if not (r['k'] > 0))}   "
      f"#(no descent direction at all, |grad|=0) = {sum(1 for r in allr if r['key'] == 'flat')}")
n4 = [r for r in good if np.isfinite(r["fl_lam"]) and r["fl_lam"] > 0
      and abs(r["d_vs"]) > 10.0 * r["fl_lam"]]
print(f"  N4 #(descent started at v* lands elsewhere than the 6-start best, >10 evaluator floors) = "
      f"{len(n4)}/{len(good)}" + (f"   worst |d_vs|/floor = {max(abs(r['d_vs']) / r['fl_lam'] for r in n4):.3e}"
                                  if n4 else ""))
n3 = [r for r in good if np.isfinite(r["fl_bp"]) and r["fl_bp"] > 0
      and abs(r["beat"] - r["bp"]) > 10.0 * r["fl_bp"]]
print(f"  N3 #(beat != g*^2/(2k*) beyond 10*floor) = {len(n3)}/{len(good)}"
      + (f"   worst = {max(abs(r['beat'] - r['bp']) / r['fl_bp'] for r in n3):.3e}" if n3 else ""))
lev = [r for r in good if np.isfinite(r["hn"]) and r["hn"] > 10.0 * r["fl_th"]]
print(f"  N5 LEVERAGE: #(... g*/k* resolved 10x above its own floor) = {len(lev)}/{len(good)}"
      f"   (b61 got 10/50; the point of moving the measurement to v* is that g*'s floor has no factor of th)")
for nm, grp in (("with leverage", lev), ("all", good)):
    if not grp:
        continue
    q = np.array([r["th"] / r["hn"] for r in grp], dtype=float)
    print(f"     {nm}: th/(g*/k*) = med {np.nanmedian(q):.4f}  min {np.nanmin(q):.4f} "
          f" max {np.nanmax(q):.4f}  n={len(grp)}")
    cb = np.array([r["beat"] / r["bp"] for r in grp if r["bp"] > 0], dtype=float)
    if cb.size:
        print(f"     {nm}: beat/(g*^2/2k*) = med {np.nanmedian(cb):.4f} "
              f" min {np.nanmin(cb):.4f}  max {np.nanmax(cb):.4f}")
print(f"  [context] m* med = {np.median([r['m'] for r in good]):.3e}  "
      f"range [{min(r['m'] for r in good):.3e},{max(r['m'] for r in good):.3e}]")
print(f"  [context] |grad_S lam_1(v*)| med = {np.median([r['nrm'] for r in allr]):.3e}")
print(f"  [context] th_meas med over all = {np.median([r['th'] for r in good]):.3e}"
      f"  max = {max(r['th'] for r in good):.3e}")
_dv = np.array([r["d_vs"] / r["gap"] for r in good], dtype=float)
print(f"  [context] N4 magnitude, (v*-descent value minus 6-start best)/gap: "
      f"med={np.nanmedian(_dv):.4e} max={np.nanmax(_dv):.4e}; median evaluator floor "
      f"fl_lam={np.nanmedian([r['fl_lam'] for r in good]):.3e} "
      f"(b60's F3 used a gap-relative criterion of 1e-6, so this floor is far finer -- a count here "
      f"means 'resolvable', not 'converged to a different basin')")

# DIAGNOSTIC ONLY (added after the run, print-only, no gate changed): where does the leverage go?
print("\n" + "=" * 132)
print("[S3] DIAGNOSTIC (print-only addition after the run, gates untouched): decompose the floors.")
print("     fl_g/|g*| is the relative precision of g* by the value route; hn/fl_th is the leverage")
print("     of the Newton step itself.")
print("=" * 132)
print(f"  {'cell':>22} {'idx':>3} {'g*':>10} {'fl_g':>10} {'fl_g/|g*|':>10} {'k*':>9} {'fl_k':>9} "
      f"{'th':>9} {'hn':>9} {'fl_th':>9} {'hn/fl_th':>9} {'beat/gap':>10} {'bp/gap':>10} {'info':>4}")
for tag, _ in CELLS:
    for i, r in enumerate(rows[tag]):
        if r["key"] == "flat":
            continue
        fl_g = r["fl_g"]
        rel = fl_g / abs(r["g"]) if r["g"] != 0.0 else float("nan")
        lev_r = r["hn"] / r["fl_th"] if (np.isfinite(r["fl_th"]) and r["fl_th"] > 0) else float("nan")
        inf = "yes" if (np.isfinite(r["hn"]) and np.isfinite(r["fl_th"]) and r["hn"] > 10.0 * r["fl_th"]) else "no"
        print(f"  {tag:>22} {i:>3} {r['g']:>10.3e} {fl_g:>10.3e} {rel:>10.3e} {r['k']:>9.3e} "
              f"{r['fl_k']:>9.3e} {r['th']:>9.3e} {r['hn']:>9.3e} {r['fl_th']:>9.3e} "
              f"{lev_r:>9.3e} {r['beat'] / r['gap']:>10.3e} {r['bp'] / r['gap']:>10.3e} {inf:>4}")

print("\n" + "=" * 132)
print("判读（预登记，跑前写下的方向；跑后只填数，不改方向）")
print("  N5 是本轮的**先决条件**：若杠杆抽样 <20/50，那么把测量挪到 v* 并没有解决 b61 的分辨率问题，")
print("     [B-CLAIM-63] 就按 53.10 的约定**停掉**，正文不写 Newton 形式，只保留 b61 的 10/50。")
print("  N1=0 且 th/(g*/k*) 的 med 在 1 附近、且**在杠杆抽样上**成立 => [B-CLAIM-63] 通过：")
print("     方向误差 = 球面上的一步 Newton，系数全部是字典可打印量（A0、B2、λ₋ 及其隐函数导数）。")
print("  N1>0 但立方修正 h_exact 通过 => claim 只能以修正形式存活，正文必须写明"
      "`纯 Newton 在 θ~1e-5 尺度上已被我的数据否证`（这条不能藏）。")
print("  N1 与 N1c 都触发 => claim 死，且因为 beat 的律是 b61 独立验过的，我会同时复查 b61。")
print("  N3 是同一件事的第二种写法（值而不是角度）：两条都通过才算这条进入正文。")
print("  N2 若 >0：那些抽样没有 Newton 方向，只能剔除并报告，不许当作通过。")
