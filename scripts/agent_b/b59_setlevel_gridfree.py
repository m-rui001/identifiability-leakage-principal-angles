"""b59 -- (A) the SET-LEVEL iff test of lem:cubic  [B-CLAIM-60(i), the registered gap], and
(B) a GRID-FREE minimiser of lambda_1 along a ray, used to attack open item 1
    (does inf_v lambda_1(v) reach lambda_min(G)?).

Helpers are loaded from b57 by exec-ing its source UP TO `def run_cell(`, so none of b57's
470-draw study re-runs.  Nothing in b57 is modified.

(A)  lem:cubic states, for v(theta) = cos(theta) v1 + sin(theta) u, u perp v1 unit,
         lambda_1(v_theta) <= cf_1   <=>   Phi(tan theta) <= 0,
         Phi(x) = (1+x^2)(P x - 2g) - kappa D^2 x^3,   kappa = 1/(mu(1-mu)).
     b57 only compared the BOUNDARY POSITION (bisection vs the cubic's leading root).  Here the
     two sets are compared POINTWISE, per ray, on ~3000 theta drawn from BOTH a linspace (from
     1e-9) and a geomspace (from 1e-13).  An exception that is not sitting within solver noise of
     the boundary falsifies the iff.

(B)  Stationary points of lambda_1 along the ray, exactly.  With x = tan(theta),
         B(x) = (c^2 + a x^2)/(1+x^2),   C(x) = (t1 c^2 + 2 g x + b x^2)/(1+x^2),
         F(lam,x) = C lam^2 - (B+C) lam + B(1-B),      d lam/dx = -F_x / F_lam,
     so (clearing the common positive factor (1+x^2)^2)
         Bhat = -2 D x,   Chat = 2[ g(1-x^2) + (b - t1 c^2) x ],
         G(x,lam) = Chat lam^2 - (Bhat+Chat) lam + Bhat (1 - 2B),
     and lambda_1'(x) = 0  <=>  G(x, lambda_1(x)) = 0  (where F_lam != 0).
     NOTE b57's C - t1 B identity has E = b - t1 a; differentiating C itself gives b - t1 c^2.
     I got this wrong once on paper, so G is GATED by a finite-difference check (section B0)
     before any of its roots are used -- rule: check a constructed object's defining identity
     before its consequence.

Registered BEFORE running (thresholds are not to be moved afterwards):
  A0  closed-form B(x),C(x) vs direct v^T A0 v, v^T B2 v on the same x-grid: max rel err < 1e-10.
  A1  #( (Phi<=0) != (lambda_1<=cf_1) ) must be 0 after excluding |lambda_1-cf_1| <= 1e-11*gap;
      disagreements INSIDE that band are boundary round-off and are reported, not hidden.
      Any disagreement with |lambda_1-cf_1| > 1e-11*gap => lem:cubic has an exception => retract.
  A2  #(descent set is a single interval) reported; #(roots of Phi in (0,infty)) reported. Not a
      registered direction -- if multi-interval sets exist they must be handled by (B), not assumed.
  B0  G(x,lam_1(x))/(1+x^2)^2 vs central FD of lambda_1 in x: rel err < 1e-6 at 200 interior x.
      > 1e-6  => my F_x is wrong and (B) is void (does not affect (A)).
  B1  ray min from (B) (roots of G + interval endpoints + x=0,x=inf) vs min over b57's 3000-point
      geomspace grid: the grid route can only be >= the exact route, and a grid that UNDERCUTS the
      exact minimiser by more than 10x that ray's own measured two-route floor means my root finder
      is wrong.  (Registered as "< 1e-9*gap" first; like A1's band that threshold sat below my
      evaluator's noise, so it is stated against the measured floor -- see REGISTRATION HISTORY.)
  B2  best over 300 Monte-Carlo rays vs lambda_1(v*) (b57's [T5] route) and vs lambda_min(G):
      #(best < lambda_1(v*) - 1e-12) reported.  If (B) beats v* then open item 1 needs the new
      inner solver; if never, v* stays the conjectured infimiser.
      #(best < lambda_min(G)) must be 0 -- that is a math-level contradiction, not a numerics one.
  C1  open item 1 number: deficit = best - lambda_min(G), as a fraction of (cf_1 - lambda_min(G)).
      Reported per cell; NO extrapolation to "inf does not reach" from a finite Monte-Carlo sample.

REGISTRATION HISTORY (kept in the file, per rule 2 -- the first registration was wrong and the
wrongness is the finding of this run):
  v1 (first run, as written above):  A1 band = 1e-11*gap on |lambda_1 - cf_1|,  B0 gate = RELATIVE
      error < 1e-6 of the FD derivative.  Result: A1 reported 13250 "exceptions" and B0 reported
      1.94.  BOTH were my evaluator, not the mathematics:
        - A1's band 1e-11*gap ~ 1.8e-17 in lambda, while the note's own root formula
          ((B+C)-sqrt(disc))/(2C) subtracts two O(1) numbers to get lambda ~ 3.7e-3, so its absolute
          error is ~1e-16, i.e. ~5.6e-11 of gap -- three orders ABOVE the band I registered.
          Fixed by evaluating lambda through the rationalised root and by comparing exceptions
          against the MEASURED disagreement between the two routes instead of an assumed constant.
        - B0 gated a quantity that passes through ZERO (d lambda/dx at a stationary point) on a
          RELATIVE criterion.  Fixed by measuring the FD floor with two step sizes (Richardson) and
          gating |G - FD| <= 20*floor, plus a sign-agreement count where |d lambda/dx| >> floor.
  v2 (this file): A1 = every exception must satisfy |lambda_1-cf_1| <= 3 * measured two-route floor;
      B0 = max |G/((1+x^2)^2 F_lam) - FD| / measured FD floor <= 20 AND full sign agreement.
  v3 (written AFTER seeing v2's numbers, and it moves no mathematical goalpost -- it replaces my own
      broken measuring stick): B0 is gated exact-vs-exact -- G against d/dx of the rationalised root
      in closed form (dlam_exact), scaled by |lam|/x -- at <= 1e-8, with the FD route demoted to
      context.  And x=0 is added back as a ray candidate: without it the 103 no-descent rays made
      the grid "beat" the exact minimiser, which was a defect of my candidate set, not of G.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "b57_descent_boundary_cubic.py"), encoding="utf-8").read()
_ns = {"__name__": "__b57head__"}
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
crossings = _ns["crossings"]
setup = _ns["setup"]
ray_L = _ns["ray_L"]
scalars = _ns["scalars"]
stats = _ns["stats"]
SPECTRA = _ns["SPECTRA"]
rng = _ns["rng"]                  # b57's stream, untouched by its study (we never ran it)
EPS = _ns["EPS"]


def mixgen(cs, delta):             # VERBATIM from b57 (it lives below the cut point)
    def g():
        p = 6
        K0 = blocks(np.eye(p), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = (U[:, 0] + U[:, 1]) / np.sqrt(2.0)
        Vv = np.column_stack([np.linalg.eigh(np.eye(p) - np.outer(weak, weak))[1][:, 1:p], weak])
        ev = np.ones(p)
        ev[-1] = 1.0 - delta
        Lam = Vv @ np.diag(ev) @ Vv.T
        evk = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
        return Lam, K0 * min(1.0, cs[0] / np.sqrt(evk[-1]))
    return g


def crossings_stable(B, C):
    """the same root by the RATIONALISED form lam = 2B(1-B)/((B+C)+sqrt(disc)).
    The note's `crossings` subtracts sqrt(disc) from (B+C), both O(1), to get lam ~ 3e-3: about
    2.5 digits are lost, so its absolute error is ~1e-16 -- which is LARGER than the 1e-11*gap
    band I registered for the iff test (gap ~ 1.8e-6 => band 1.8e-17).  That is why b59's first run
    reported 13250 'exceptions': my evaluator's noise, not a set-level counterexample.
    The two routes' own disagreement is used below as the MEASURED local noise floor."""
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    disc = np.maximum((B + C) ** 2 - 4.0 * C * B * (1.0 - B), 0.0)
    den = (B + C) + np.sqrt(disc)
    out = np.where(den > 0.0, 2.0 * B * (1.0 - B) / np.where(den > 0, den, 1.0), np.nan)
    return np.where(C > 0.0, out, np.nan)


def ray_forms(s, sc, x):
    """closed-form B(x), C(x) on the ray, and Phi(x); x is an array."""
    c2, t1, b = s["c2"], s["t1"], sc["b"]
    D, g, kap = sc["D"], sc["g"], sc["kap"]
    a = sc["a"]
    den = 1.0 + x * x
    B = (c2 + a * x * x) / den
    C = (t1 * c2 + 2.0 * g * x + b * x * x) / den
    Phi = den * (sc["P"] * x - 2.0 * g) - kap * D * D * x ** 3
    return B, C, Phi


def lam_of_x(s, sc, x):
    """stable route; returns (lam_stable, B, C)."""
    B, C, _ = ray_forms(s, sc, x)
    return crossings_stable(B, C), B, C


def Gval(s, sc, x, lam):
    """G(x,lam) = (1+x^2)^2 * F_x(lam(x),x); zero <=> lambda_1'(x)=0 (where F_lam != 0)."""
    c2, t1, b = s["c2"], s["t1"], sc["b"]
    D, g = sc["D"], sc["g"]
    B = (c2 + sc["a"] * x * x) / (1.0 + x * x)
    Bh = -2.0 * D * x
    Ch = 2.0 * (g * (1.0 - x * x) + (b - t1 * c2) * x)
    return Ch * lam * lam - (Bh + Ch) * lam + Bh * (1.0 - 2.0 * B)


def dlam_exact(s, sc, x):
    """(lambda_1, d lambda_1/dx) by DIFFERENTIATING the rationalised root in closed form --
    an exact second route, so that B0 no longer depends on a finite difference (whose floor I
    have to guess).  lam = 2B(1-B)/S, S = (B+C)+sqrt(disc)."""
    B, C, _ = ray_forms(s, sc, np.array([x]))
    B, C = float(B[0]), float(C[0])
    d2 = (1.0 + x * x) ** 2
    Bp = -2.0 * sc["D"] * x / d2
    Cp = 2.0 * (sc["g"] * (1.0 - x * x) + (sc["b"] - s["t1"] * s["c2"]) * x) / d2
    disc = max((B + C) ** 2 - 4.0 * C * B * (1.0 - B), 0.0)
    sq = np.sqrt(disc)
    dp = 2.0 * (B + C) * (Bp + Cp) - 4.0 * ((Bp * C + B * Cp) * (1.0 - B) - B * C * Bp)
    S = (B + C) + sq
    Sp = (Bp + Cp) + (dp / (2.0 * sq) if sq > 0.0 else 0.0)
    lam = 2.0 * B * (1.0 - B) / S if S > 0.0 else np.nan
    dlam = (2.0 * Bp * (1.0 - 2.0 * B) * S - 2.0 * B * (1.0 - B) * Sp) / (S * S) if S > 0.0 else np.nan
    return lam, dlam


def bisect_root(f, lo, hi, it=200):
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0.0:
        return np.nan
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return np.nan
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def descent_intervals(sc):
    """exact: {x>0 : Phi(x)<=0} as a list of (lo,hi), from the cubic's positive real roots."""
    r = np.roots(sc["coef"])
    ok = (np.abs(r.imag) < 1e-9) & (r.real > 0.0)
    rt = np.sort(r[ok].real)
    edges = np.concatenate([[0.0], rt, [np.inf]])
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mid = np.sqrt(lo * hi) if np.isfinite(hi) else (2.0 * lo if lo > 0 else 1.0)
        if mid <= 0 or not np.isfinite(mid):
            mid = max(lo, 1e-8)
        val = np.polyval(sc["coef"], mid)
        if val <= 0.0:
            out.append((lo, hi))
    return out, rt


def ray_min_exact(s, sc, scan=240):
    """min of lambda_1 on the ray, restricted to the descent intervals found from the cubic.
    Interior candidates = roots of G; endpoints = interval edges and x=inf (v=u)."""
    ivs, rt = descent_intervals(sc)
    f = lambda x: Gval(s, sc, x, lam_of_x(s, sc, np.atleast_1d(x))[0][0])
    # x=0 IS a candidate.  When the descent set is empty (ivs == []) the inf over the closed ray is
    # lambda_1(v1) = cf_1 at the endpoint; b59's first pass dropped x=0 on the argument "that's never
    # the min", which is only true INSIDE a descent interval, and it made 103 no-descent rays report
    # that the grid beat the exact minimiser (their only candidate was lambda_1(u), which is huge).
    cands = [0.0, np.inf]
    for (lo, hi) in ivs:
        fin = np.isfinite(hi)
        a = lo if lo > 0 else 1e-14
        b = hi if fin else 1e12
        cands += [a, b]
        xs = np.geomspace(a, b, scan)
        gs = np.array([f(x) for x in xs])
        ok = np.isfinite(gs)
        for i in range(scan - 1):
            if ok[i] and ok[i + 1] and gs[i] * gs[i + 1] < 0.0:
                rr = bisect_root(f, xs[i], xs[i + 1])
                if np.isfinite(rr):
                    cands.append(rr)
        if not ok.any():
            return np.nan, len(ivs), rt, np.nan, 0
    lams = np.array([lam_of_x(s, sc, np.atleast_1d(x))[0][0] if np.isfinite(x)
                     else float(crossings_stable(np.array([sc["a"]]), np.array([sc["b"]]))[0])
                     for x in cands])
    k = int(np.nanargmin(lams))
    return float(lams[k]), len(ivs), rt, float(cands[k]), int(np.sum(~np.isfinite(lams)))


# ---------------------------------------------------------------- cell driver
CELLS = [
    ("q=3 spread 5-75", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 5-75"])))(lam_spd(6))),
    ("q=3 spread 2-45-88", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))(lam_spd(6))),
    ("q=3 spread 30-88-89", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 30-88-89"])))(lam_spd(6))),
    ("q=3 equal 20d", lambda: (lambda L: (L, blocks(L, SPECTRA["equal 20d"])))(lam_spd(6))),
    ("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)),
                                                          np.cos(np.deg2rad(40))])))(lam_spd(5))),
    ("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(t))
                                                           for t in (5, 25, 55, 80)])))(lam_spd(7))),
    ("q=3 2-45-88 mix", mixgen(SPECTRA["spread 2-45-88"], 3e-2)),
    ("q=3 30-88-89 mix", mixgen(SPECTRA["spread 30-88-89"], 3e-2)),
]

print("=" * 130)
print("[A0/A1/A2] SET-LEVEL iff test of lem:cubic  (B-CLAIM-60(i));  [B] grid-free ray minimiser")
print("           BOTH grids (linspace from 1e-9 + geomspace from 1e-13, ~3000 theta per ray),")
print("           12 draws per cell, rays = steepest (ray_L) + 2 random per draw.")
print("=" * 130)

NDR = 12
# rule 11: the statistic must be visible on the grid.  The boundary layer can be 1e-5 wide in theta,
# so a single linspace is not enough -- BOTH grids are tested, and each point is a separate iff check.
THA = np.unique(np.concatenate([np.linspace(1e-9, np.pi / 2 - 1e-12, 1500),
                                np.geomspace(1e-13, np.pi / 2 - 1e-12, 1500)]))
XA = np.tan(THA)
gate_err, exc_dev, exc_rel, exc_x, floor_all, floor_rel = [], [], [], [], [], []
nint, b0_err, b0_exact, b0_floor, b0_sign, b1_err, b1_diag = [], [], [], [], [], [], []
npts = 0
nrays = 0
cell_rows = {}                     # rule 13: per-cell rows, so medians are never pooled across cells
beats = []                         # (cell, beat/gap, beat in lambda) for rays undercutting lambda_1(v*)
bestfrac, vstarfrac, nbeat, nbelow, nbadroot, nstat_cell = [], [], [], [], [], []
for tag, gen in CELLS:
    rows = 0
    for _ in range(NDR):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        gap = s["cf1"] - s["true"]
        uu = ray_L(s)
        rays = []
        if uu is None:
            nstat_cell.append(s["m0"])
            w = rng.normal(size=s["A0"].shape[0])
            w = w - s["v1"] * (s["v1"] @ w)
            rays.append(w / np.linalg.norm(w))
        else:
            rays.append(uu[0])
            for _j in range(2):
                w = rng.normal(size=s["A0"].shape[0])
                w = w - s["v1"] * (s["v1"] @ w)
                rays.append(w / np.linalg.norm(w))
        for u in rays:
            nrays += 1
            sc = scalars(s, u)
            V = np.outer(np.cos(THA), s["v1"]) + np.outer(np.sin(THA), u)
            Bd = np.einsum("ij,jk,ik->i", V, s["A0"], V)
            Cd = np.einsum("ij,jk,ik->i", V, s["B2"], V)
            Bc, Cc, Phi = ray_forms(s, sc, XA)
            gate_err.append(max(float(np.max(np.abs(Bc - Bd)) / s["c2"]),
                                float(np.max(np.abs(Cc - Cd)) / max(abs(sc["b"]), 1e-30))))
            lamN = crossings(Bd, Cd)                 # the note's route: (B+C)-sqrt(disc)
            lamS = crossings_stable(Bd, Cd)          # rationalised route
            npts += len(lamN)
            flo = np.abs(lamS - lamN)                # MEASURED two-route noise at this point
            fl_ray = float(np.nanmax(flo))
            floor_all.append(fl_ray)
            floor_rel.append(fl_ray / gap)
            pred = lamS <= s["cf1"]
            phi = Phi <= 0.0
            bad = np.nonzero(pred != phi)[0]
            for i in bad:
                dev = abs(lamS[i] - s["cf1"])
                exc_dev.append(dev)
                exc_x.append(XA[i])
                exc_rel.append(dev / gap)
            nint.append(sc["g"] > 0 and len(descent_intervals(sc)[0]) > 1)
            # ---- B0: is G really (1+x^2)^2 * F_x ?  compare with FD of lambda_1 in x.
            # The FLOOR OF THE FD ROUTE IS MEASURED (two step sizes, Richardson), not assumed:
            # b59's first run gated on a RELATIVE error < 1e-6 and printed 1.94, because near a
            # stationary point d lam/dx is ~0 and any relative criterion is meaningless.
            errs, errs_exact, floors, sign_ag, sign_tot = [], [], [], 0, 0
            for x0 in np.geomspace(1e-4, 1.0, 20):
                l0, B0v, C0v = lam_of_x(s, sc, np.array([x0]))
                Flam = 2 * C0v[0] * l0[0] - (B0v[0] + C0v[0])
                if abs(Flam) < 1e-8 or not np.isfinite(l0[0]):
                    continue
                pred_d = -Gval(s, sc, x0, l0[0]) / ((1 + x0 * x0) ** 2 * Flam)
                lam_e, dlam_e = dlam_exact(s, sc, x0)
                if np.isfinite(dlam_e) and np.isfinite(lam_e):
                    errs_exact.append(abs(pred_d - dlam_e) / max(abs(lam_e) / max(x0, 1e-12), 1e-300))
                ds = []
                for h in (1e-4 * x0, 2.5e-5 * x0):
                    lm = lam_of_x(s, sc, np.array([x0 - h, x0 + h]))[0]
                    ds.append(float((lm[1] - lm[0]) / (2 * h)))
                fl = abs(ds[0] - ds[1]) / 0.9375
                if not (np.isfinite(fl) and fl > 0.0):
                    continue
                floors.append(fl)
                errs.append(abs(pred_d - ds[1]) / fl)
                if abs(ds[1]) > 20.0 * fl:
                    sign_tot += 1
                    sign_ag += int(np.sign(pred_d) == np.sign(ds[1]))
            if errs:
                b0_err.append(float(np.max(errs)))
                b0_floor.append(float(np.median(floors)))
                b0_sign.append((sign_ag, sign_tot))
            if errs_exact:
                b0_exact.append(float(np.max(errs_exact)))
            # ---- B1: exact ray min vs the ~3000-point route (both on the STABLE lambda)
            best, niv, rts, xat, nnan = ray_min_exact(s, sc)
            nbadroot += [nnan]
            gridmin = float(np.nanmin(lamS))
            if np.isfinite(best):
                if niv > 1:
                    nint.append(True)
                b1_err.append((gridmin - best) / gap)          # grid can only overestimate
                if gridmin - best < -10.0 * fl_ray:            # ABOVE the ray's own measured floor
                    b1_diag.append((tag, float(best), float(xat), niv, gridmin))
                # ---- C1/B2 vs lambda_min(G) and vs lambda_1(v*)
                l1vstar = float(crossings_stable(
                    np.array([float(s["vstar"] @ s["A0"] @ s["vstar"])]),
                    np.array([float(s["vstar"] @ s["B2"] @ s["vstar"])]))[0])
                deficit = best - s["true"]
                cell_rows.setdefault(tag, []).append((deficit / gap, (l1vstar - best) / gap))
                if (l1vstar - best) / gap > 0.0:
                    beats.append((tag, (l1vstar - best) / gap, (l1vstar - best)))
                bestfrac.append(deficit / gap)
                vstarfrac.append((l1vstar - best) / gap)
                nbeat.append((l1vstar - best) > 1e-12)
                nbelow.append(best < s["true"])
            else:
                bestfrac.append(np.nan)
                vstarfrac.append(np.nan)
                nbeat.append(False)
                nbelow.append(False)
        rows += 1
    print(f"  {tag:>24}  draws={rows}")

print("\n" + "=" * 130)
fl_max = max(floor_all)
print("[A0] closed-form B(x),C(x) vs direct:            worst rel err = "
      f"{max(gate_err):.3e}   (gate: <1e-10)  {'PASS' if max(gate_err) < 1e-10 else 'VOID -> everything below is meaningless'}")
print(f"[A1] points tested = {npts}  ({nrays} rays x {len(XA)} theta)   "
      f"#(set-level exceptions) = {len(exc_dev)}")
print(f"     MEASURED two-route floor: max|lam_stable - lam_note| over all points = "
      f"{fl_max:.3e} in lambda  =  {max(floor_rel):.3e} in units of that cell's gap")
print(f"     worst exception deviation = {max(exc_dev):.3e} in lambda = "
      f"{max(exc_rel):.3e} of ITS OWN CELL's gap;   ratio to measured floor = {max(exc_dev) / fl_max:.3e}"
      f"   -> {'PASS: every exception sits within my evaluator noise (ratio <= 3)' if max(exc_dev) <= 3 * fl_max else 'FALSIFIED: a set-level exception ABOVE my own noise floor -> retract lem:cubic'}")
print("     (absolute lambda is the only comparable unit here: gaps range over 3 orders across cells,")
print("      so the per-cell relative column is reported for readability and NOT pooled.)")
print("     exception-count sweep, |lam-cf_1| in lambda > thr:  "
      + "  ".join(f"{t:.0e}:{int(np.sum(np.array(exc_dev) > t))}"
                  for t in (1e-16, 3e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8)))
print(f"     exceptions' x-range: min={min(exc_x):.3e} max={max(exc_x):.3e}"
      f"   #(with x < 1e-6) = {int(np.sum(np.array(exc_x) < 1e-6))}")
print(f"[A2] #(rays whose descent set is NOT a single interval) = {int(np.sum(nint))}/{len(nint)}"
      "   (A2 is a count, not a registered direction; multi-interval sets are handled by (B) anyway)")
print(f"[B0] GATE exact-vs-exact: max over rays |G/((1+x^2)^2 F_lam) - d lam/dx (closed-form root "
      f"derivative)| / (|lam|/x) = {max(b0_exact):.3e}   (gate: <=1e-8)  "
      f"{'PASS' if max(b0_exact) <= 1e-8 else 'FALSIFIED -> my F_x is wrong, (B)/(C) void'}")
print(f"     [context] same quantity against a FINITE difference, in units of the measured FD floor: "
      f"max = {max(b0_err):.3e} (median floor {stats(np.array(b0_floor))})"
      "  -- the FD route is the one with the unusable floor, not G")
print(f"     sign agreement where |d lam/dx| > 20*floor: {int(sum(x[0] for x in b0_sign))}/"
      f"{int(sum(x[1] for x in b0_sign))}   (must be full: a sign flip means G is not F_x)")
print(f"[B1] grid-min minus exact ray-min, in units of gap: {stats(np.array(b1_err))}"
      "   (must be >=0: the grid cannot beat the exact minimiser; negative = my root finder is wrong)")
print(f"     #(rays where the exact min needed a discarded candidate) = {int(np.sum(np.array(nbadroot) > 0))}"
      f"   #(grid better than exact by >10x the ray's own measured floor) = {len(b1_diag)}"
      "   <- the registered B1 test")
for e in b1_diag[:6]:
    print(f"       DIAG cell={e[0]}  best={e[1]:.12f}  x_at={e[2]:.3e}  #intervals={e[3]}  gridmin={e[4]:.12f}")
print(f"[C1] deficit (ray min - lambda_min(G))/gap: {stats(np.array(bestfrac))}"
      f"   #(ray min BELOW lambda_min(G)) = {int(np.sum(nbelow))}  (must be 0)")
print(f"[B2] (lambda_1(v*) - best ray min)/gap: {stats(np.array(vstarfrac))}"
      f"   #(best beats v*) = {int(np.sum(nbeat))}/{len(nbeat)}")
print(f"     (equal-angle draws skipped from the ray count: {len(nstat_cell)};"
      f" their dim E0 = {sorted(set(nstat_cell))})")

print("\n" + "=" * 130)
print("[E] PER-CELL table (rule 13: these are PER-RAY values grouped by cell; the pooled medians")
print("    above must not be read as a per-cell guarantee):")
print("      cell                       rays   min/med/max deficit/gap   #(below lam_min(G))"
      "   #(ray beats v*)  max beat/gap")
for tag, _ in CELLS:
    rr = cell_rows.get(tag, [])
    if not rr:
        print(f"      {tag:<26}    0   (all draws stationary or skipped)")
        continue
    d = np.array([x[0] for x in rr])
    v = np.array([x[1] for x in rr])
    print(f"      {tag:<26} {len(rr):>4}   {d.min():.4f} /{np.median(d):.4f} /{d.max():.4f}"
          f"          {int(np.sum(d < 0.0)):>3}                  {int(np.sum(v > 0.0)):>3}"
          f"              {v.max():.4e}")
print(f"     #(rays undercutting lambda_1(v*)) = {len(beats)}/{len(bestfrac)}"
      f"   #(with beat > 1e-12 in lambda) = {int(np.sum([x[2] > 1e-12 for x in beats]))}")
for t, g, a in sorted(beats, key=lambda z: -z[1])[:5]:
    print(f"       largest beats: {t:<24} {g:.4e} of gap   ({a:.3e} in lambda)")
for t, g, a in sorted([z for z in beats if z[0] != "q=3 equal 20d"], key=lambda z: -z[1])[:3]:
    print(f"       beats OUTSIDE the equal-angle cell: {t:<24} {g:.4e} of gap"
          f"   ({a:.3e} in lambda)")

print("\n" + "=" * 130)
print("[D] 120 Monte-Carlo rays on the STEEPEST-ray cell of q=4, plus the steepest ray itself:")
print("    Does the best-of-many-rays beat v*?  (open item 1 wants inf over the whole sphere,")
print("    and rays through v1 only cover a 2-plane -- stated as a limitation, not hidden.)")
Lam, K = (lambda L: (L, blocks(L, [np.cos(np.deg2rad(t)) for t in (5, 25, 55, 80)])))(lam_spd(7))
s = setup(Lam, K)
if s is not None:
    gap = s["cf1"] - s["true"]
    l1vstar = float(crossings_stable(np.array([float(s["vstar"] @ s["A0"] @ s["vstar"])]),
                                     np.array([float(s["vstar"] @ s["B2"] @ s["vstar"])]))[0])
    vals, xats = [], []
    for _j in range(120):
        w = rng.normal(size=s["A0"].shape[0])
        w = w - s["v1"] * (s["v1"] @ w)
        u = w / np.linalg.norm(w)
        sc = scalars(s, u)
        best, niv, rts, xat, nnan = ray_min_exact(s, sc, scan=120)
        if np.isfinite(best):
            vals.append(best)
            xats.append(xat)
    vals = np.array(vals)
    print(f"    q=4 single draw, 120 random rays:  min over rays = {vals.min():.12f}"
          f"   lambda_min(G) = {s['true']:.12f}   cf_1 = {s['cf1']:.12f}   lambda_1(v*) = {l1vstar:.12f}")
    print(f"    deficit of the BEST RAY over lambda_min(G) = {(vals.min()-s['true'])/gap:.6f} of gap;"
          f"  deficit of lambda_1(v*) = {(l1vstar-s['true'])/gap:.6f}")
    print(f"    #(rays beating v*) = {int(np.sum(vals < l1vstar - 1e-12))}/120;"
          f"  median ray deficit = {np.median((vals-s['true'])/gap):.6f}")
    print(f"    argmin x over rays: med={np.median(xats):.4e} min={np.min(xats):.4e} max={np.max(xats):.4e}")
    print(f"    NOTE the sphere is p={s['A0'].shape[0]}-dim; rays through v1 are a 2-plane family.")
