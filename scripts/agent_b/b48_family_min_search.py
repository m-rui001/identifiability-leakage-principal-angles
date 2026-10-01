"""
b48 -- B-CLAIM-54(a): is cf1 the SMALLEST member of the chord family lam(v) of (20)?

Why this is the right next question.  prop:chord proves lam_min(G) <= lam(v) for EVERY unit v, and
cf1 is one named member (v1 in E0 maximising C).  b46/b47 found no direction with lam(v) < cf1, but
they used 200 random directions per draw in a sphere of dimension q-1 >= 2 -- sampling, not searching.
I made exactly that mistake once already (b36's mu(w0) conjecture, killed in b37 by a DENSE CIRCLE
SEARCH), so this script searches instead: a polar grid AROUND v1,

      v(t,u) = cos(t) v1 + sin(t) u,   u in S^{q-2} perpendicular to v1,  t in [0, pi],

which covers the whole sphere S^{q-1} exactly (for q=3 the perpendicular sphere is a circle and
(t,u) is polar coordinates on S^2; for q=4 it is S^2 x [0,pi]).  Searching around v1 is the only
place a competitor could hide, since lam(v1) = cf1 and t=0 recovers it.

The claim reduces to ONE inequality, which is why both are reported:
  lam(v) >= cf1 for all v      <=>      s_v(cf1) <= s_{v1}(cf1) for all v with B(v) - cf1 C(v) > 0,
because s_v is increasing with a single crossing against the decreasing line 1-lam, and a direction
whose surrogate already blew up before cf1 (B(v)-cf1 C(v) <= 0) would cross EARLIER, i.e. beat cf1.
So the search prints, per cell:
  (i)  min_v lam(v) - cf1                     [predicted >= 0; a resolved negative => cf1 NOT optimal]
  (ii) # {v : B(v) - cf1*C(v) <= 0}           [predicted 0; any hit is a competitor by (i)'s logic]
  (iii)# {v : lam(v) < lam_min(G) - floor}    [MUST be 0 -- that is prop:chord itself, the
                                               implementation audit of the sharpened claim]
  (iv)  the intercept ratio B(v*)/c^2 at the minimiser v*, and its |sin angle| to E0
       [if a competitor exists it will show up here: B < c^2 means the smaller intercept bought a
        lower crossing by giving up height, which is exactly the trade cf1 does NOT make]

Pre-registered, before running:
  A  min_v lam(v) == cf1 (to grid resolution) in every cell, so cf1 is the family minimum; the
     residual min_v lam - cf1 >= -floor and the minimiser sits at t = 0 (i.e. v* = v1).
  B  if instead some cell gives min_v lam(v) < cf1 - floor, then a TIGHTER provably one-sided
     surrogate exists (the family minimum), and I must report the ratio cf1/min_v and where it wins;
     that would upgrade the open item into a real sharpening, not a cosmetic one.
  C  (iii) must be zero everywhere.  A single resolved violation would mean prop:chord is WRONG, in
     which case nothing else in this file matters and the chapter must be re-opened.
  D  at the exact loci (isotropy ZK=0, q=1) cf1 = lam_min, so min_v lam(v) >= cf1 with the minimiser
     AT v1 and no gap: the search there is a control, not a test.

WHAT I GOT WRONG IN THE FIRST VERSION (recorded, as in b47): I took v1 = E0[:,-1], an arbitrary
eigenvector of the compression.  B(v) = c^2 for every unit v in E0, but the C-MAXIMISER inside a
degenerate E0 is the top eigenvector of E0^T B2 E0, not of A0; with dim E0 = q (equal angles) the two
are unrelated.  The self-check 'max|d0|' existed precisely to catch this and it did: 1.38e-02 in the
equal-20d cell, i.e. that cell's 'competitor' was my own wrong direction, not a real one.  Fixed to
v1 = E0 @ W[:, -1], and dim E0 is now printed per cell rather than assumed.

MECHANISM (derived after that first run and BEFORE the corrected one, so it is a prediction with a
falsifier).  F(lam;B,C) = C lam^2 - (B+C) lam + B(1-B) has F_lam = -sqrt(disc) < 0 at the SMALLER
root, hence
      dlam/dC = -F_C/F_lam = lam(lam-1)/|F_lam| < 0,     dlam/dB = (1-lam-2B)/|F_lam|,
and the second is negative once B > (1-lam)/2.  t1 maximises C over E0 ONLY, so a direction off v1
can buy C(v) > c^2 t1 by paying B(v) < c^2, and that trade LOWERS the crossing whenever the C-gain
dominates the B-loss.  Prediction for the corrected run: every competitor has Cratio = C(v*)/(c^2 t1)
> 1 together with Bratio < 1.  Falsifier: a competitor with Cratio <= 1 => the trade is not the
mechanism and my derivative signs are wrong.
"""
import numpy as np

rng = np.random.default_rng(50126)
N = 40
EPS = np.finfo(float).eps
NT = 96            # polar angle grid, t in (0, pi)
NU = 240           # points on the perpendicular sphere S^{q-2}

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
    """vectorised smaller root of C lam^2 - (B+C) lam + B(1-B) = 0 (the family (20))."""
    disc = (Bs + Cs) ** 2 - 4.0 * Cs * Bs * (1.0 - Bs)
    bad = disc < 0.0
    disc = np.where(bad, 0.0, disc)
    out = np.where(Cs > 0.0, ((Bs + Cs) - np.sqrt(disc)) / (2.0 * np.where(Cs > 0, Cs, 1.0)), 1.0)
    return np.where(bad, 1.0, out)


def perp_sphere(v1):
    """a net on S^{q-2} = the unit sphere of v1^perp, returned as (NU, q) ROWS OF UNIT VECTORS."""
    q = v1.size
    U, s, Vt = np.linalg.svd(np.eye(q) - np.outer(v1, v1))
    B = Vt[: q - 1].T                      # (q, q-1): orthonormal basis of v1^perp
    if q == 2:                             # S^0 = {+b, -b}: one row, and the t-grid must go 0..2pi
        b = B[:, 0]
        return b[None, :]
    if q == 3:                             # circle in the 2-dim perpendicular plane
        u1, u2 = B[:, 0], B[:, 1]
        th = np.linspace(0.0, 2 * np.pi, NU, endpoint=False)
        return np.outer(np.cos(th), u1) + np.outer(np.sin(th), u2)
    # q >= 4: random unit rows -- a sample of S^{q-2}, NOT a covering net (recorded as a limitation)
    R = rng.normal(size=(NU, q - 1))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R @ B.T


def setup(Lam, K):
    """everything [E] and the search need, in ONE place so the two cannot disagree."""
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
    m0 = E0.shape[1]
    mu, W = np.linalg.eigh(E0.T @ B2 @ E0)          # C restricted to E0: A0 v = c^2 v there, so
    t1 = float(mu[-1]) / c2                          # B(v) = c^2 for EVERY unit v in E0
    cf1 = cf_of(t1, c2)
    v1 = E0 @ W[:, -1]                               # THE direction attaining t1 -- NOT E0[:, -1],
    v1 = v1 / np.linalg.norm(v1)                     # which is arbitrary once dim E0 > 1 (see below)
    lbar = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    if cf1 >= lbar:
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, t1=t1, cf1=cf1, v1=v1, m0=m0, lbar=lbar,
                floor=100.0 * EPS * np.linalg.norm(Gr, 2))


def search(Lam, K):
    s = setup(Lam, K)
    if s is None:
        return None
    q = s["v1"].size
    c2, cf1, true, v1, floor = s["c2"], s["cf1"], s["true"], s["v1"], s["floor"]
    A0, B2, m0, t1 = s["A0"], s["B2"], s["m0"], s["t1"]
    # t = 0 IS included: the point v1 is the claimed minimiser, so the net must contain it exactly
    # (otherwise 'min_v lam(v) - cf1' is a grid-resolution statement, not a comparison with cf1).
    top = 2.0 * np.pi if q == 2 else np.pi - 1e-3      # q=2: the whole circle comes from t alone
    tt = np.concatenate(([0.0], np.linspace(1e-3, top, NT - 1)))
    Us = perp_sphere(v1)
    Vs = (np.cos(tt)[:, None] * v1[None, :])[:, None, :] + (np.sin(tt)[:, None, None] * Us[None, :, :])
    Vs = Vs.reshape(-1, q)
    uerr = float(np.max(np.abs(np.linalg.norm(Vs, axis=1) - 1.0)))   # the net must be ON the sphere
    Bs = np.einsum("ij,jk,ik->i", Vs, A0, Vs)
    Cs = np.einsum("ij,jk,ik->i", Vs, B2, Vs)
    ls = crossings(Bs, Cs)
    blow = Bs - cf1 * Cs <= 0.0
    ring = np.repeat(np.arange(tt.size), Us.shape[0])     # ring 0 = v1 itself, repeated NU times
    kl = int(np.argmin(np.where(ring > 0, ls, np.inf)))   # best COMPETITOR, off the claimed minimiser
    it = ring[kl]
    sinang = float(np.linalg.norm(Vs[kl] - v1 * (v1 @ Vs[kl])))
    # FRACTION of cf1's over-estimate that the best direction removes: 0 = cf1 is the family min,
    # 1 = the family min is exact, negative = the net found nothing below cf1.
    red = float(1.0 - (ls[kl] - true) / (cf1 - true)) if cf1 > true + floor else np.nan
    return dict(true=true, cf1=cf1, floor=floor, m0=m0, uerr=uerr,
                d0=ls[0] - cf1,                           # v1 self-check: must be round-off
                d_pos=ls[kl] - ls[0],                     # < 0 by >floor => competitor (claim B)
                tstar=float(tt[it]), frac=float(it) / float(tt.size - 1), red=red,
                n_blow=int(blow.sum()), n=len(ls),
                viol=int(np.sum((ls < true) & (true - ls > floor))),
                Bratio=float(Bs[kl] / c2), Cratio=float(Cs[kl] / (c2 * t1)),
                sinang=sinang, gap1=cf1 - true)


def mixgen(cs, delta):
    def g():
        K0 = blocks(np.eye(6), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = (U[:, 0] + U[:, 1]) / np.sqrt(2.0)
        Vv = np.column_stack([np.linalg.eigh(np.eye(6) - np.outer(weak, weak))[1][:, 1:6], weak])
        ev = np.ones(6)
        ev[-1] = 1.0 - delta
        Lam = Vv @ np.diag(ev) @ Vv.T
        evk = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
        return Lam, K0 * min(1.0, cs[0] / np.sqrt(evk[-1]))
    return g


def cell(tag, gen, n=N):
    recs = []
    for _ in range(n):
        rec = search(*gen())
        if rec is not None:
            recs.append(rec)
    if not recs:
        print(f"   {tag:>30}: all skipped")
        return
    d0 = np.array([r["d0"] for r in recs])
    dp = np.array([r["d_pos"] for r in recs])
    fl = np.array([r["floor"] for r in recs])
    rd = np.array([r["red"] for r in recs])
    res = np.abs(dp) > fl                       # competitors that clear the draw's own floor
    nb = int(np.sum(res & (dp < 0.0)))
    br_ = np.array([r["Bratio"] for r in recs]); sa_ = np.array([r["sinang"] for r in recs])
    ts_ = np.array([r["tstar"] for r in recs]);  fa_ = np.array([r["frac"] for r in recs])
    print(f"   {tag:>30}: n={len(recs):3d} dimE0={sorted({r['m0'] for r in recs})}"
          f"  v1 self-check max|d0|={np.abs(d0).max():.2e}"
          f"  competitor d: max={dp.max():9.2e} min={dp.min():9.2e}"
          f"  #below v1>floor={nb}/{int(res.sum())}"
          f"  #blow-up={sum(r['n_blow'] for r in recs)} viol(prop:chord)={sum(r['viol'] for r in recs)}"
          f"  max| |v|-1 |={max(r['uerr'] for r in recs):.1e}")
    print(f"{'':>34}  at v*: B/c^2={np.median(br_):6.3f} C/(c^2 t1)={np.median(np.array([r['Cratio'] for r in recs])):6.3f}"
          f" |sin to v1|={np.median(sa_):6.3f}"
          f" t*={np.median(ts_):5.2f} (ring {np.median(fa_)*100:4.1f}% of grid)"
          f"  frac of cf1 gap removed: med={(np.nanmedian(rd) if np.isfinite(rd).any() else float('nan')):6.3f}"
          f" max={(np.nanmax(rd) if np.isfinite(rd).any() else float('nan')):6.3f}"
          f"  med gap1={np.median([r['gap1'] for r in recs]):9.2e}")
    bad = [r for r, d in zip(recs, dp) if d < 0.0 and abs(d) > r["floor"] and r["Cratio"] <= 1.0]
    if bad:
        print(f"{'':>34}  FALSIFIER of the (B,C) trade: {len(bad)} competitor(s) with C(v*) <= c^2 t1")
    comp = [r for r, d in zip(recs, dp) if d < 0.0 and abs(d) > r["floor"]]
    if comp:   # the trade happens at O(t^2) in B and O(t) in C, so 3 decimals cannot show it
        print(f"{'':>34}  competitors (n={len(comp)}): B/c^2 in [{min(r['Bratio'] for r in comp):.6e},"
              f"{max(r['Bratio'] for r in comp):.6e}]  C/(c^2 t1) in "
              f"[{min(r['Cratio'] for r in comp):.6e},{max(r['Cratio'] for r in comp):.6e}]")


print("=" * 118)
print("[A/B/C/D] exhaustive polar search around v1 over the whole unit sphere, q=3 and q=2 and q=4.")
print("    The net INCLUDES t=0, i.e. v1 itself: 'max|d0|' checks that lam(v1) reproduces cf1, then")
print("    'best competitor' compares every other net point against lam(v1), not against a grid hint.")
print("    (i) #below v1 by >floor must be 0 for every draw  => cf1 IS the family minimum (claim A).")
print("    (ii) #blow-up-before-cf1 must be 0: a direction whose chord blows up before cf1 would")
print("         cross earlier and beat cf1 by construction, so this count IS the competitor detector.")
print("    (iii) viol = #{lam(v) < lam_min(G)} must be 0 -- that is prop:chord, re-audited on a dense net.")
print("    (iv) where the competitor sits: B(v*)/c^2 (1.000 means v* is in E0) and |sin| to v1.")
for name, cs in SPECTRA.items():
    cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix"),
                      ([np.cos(np.deg2rad(2)), np.cos(np.deg2rad(84)), np.cos(np.deg2rad(88))], 3e-2,
                       "q=3 2d/84d/88d mix")]:
    cell(nm, mixgen(cs, delta))
cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
cell("q=2 equal 5d", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5))] * 2)))(lam_spd(5)))
cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))
print("   [D] control: isotropy ZK=0 (cf1 exact, so the family minimum must sit AT v1 with no slack)")
def isogen(cs, delta, seed):
    def g():
        p = 6
        K0 = blocks(np.eye(p), cs)
        Q, _ = np.linalg.qr(rng.normal(size=(p, p - len(cs))))
        w = Q[:, 0]
        w = w / np.linalg.norm(w)
        Lam = np.eye(p) - delta * np.outer(w, w)
        return Lam, K0
    return g
for cs, nm in [(SPECTRA["spread 2-45-88"], "iso ZK=0 spread"), (SPECTRA["equal 20d"], "iso ZK=0 equal")]:
    cell(nm, isogen(cs, 1e-3, 0))

print("\n" + "=" * 118)
print("[E] IS v1 STATIONARY for lam on the sphere?  the ORDER of the dip decides what cf1 is.")
print("    Along v(t) = cos t v1 + sgn sin t u with u _|_ v1:")
print("      B'(0) = 2 v1^T A0 u = 2 c^2 (v1.u) = 0   (A0 v1 = c^2 v1: v1 is a top eigenvector)")
print("      C'(0) = 2 sgn v1^T B2 u  = 2 sgn |P_perp B2 v1|   for u along that component, and")
print("      dlam/dC = lam(lam-1)/|F_lam| < 0  =>  lam'(0+) = C'(0) lam(lam-1)/|F_lam|.")
print("    PREDICTION (before running): |P_perp B2 v1| > 0 whenever dim E0 < q, because t1 maximises C")
print("    over E0 ONLY -- so v1 is a SADDLE, not a minimum, and cf1 is beaten at FIRST order along")
print("    the steepest-descent ray, with lam'(0+) matching the formula above; the dip must then")
print("    saturate at t* = O(|slope|/curvature) and prop:chord must still hold along the ray")
print("    (viol = 0).  In the degenerate cells (dim E0 = q: equal angles) B2 v1 = lam_max(B2) v1 for")
print("    the C-maximising v1, so the slope is 0 and NO competitor exists -- [A] already showed 0/40")
print("    there, which is the consistency check this section is designed to fail on.  Falsifier of")
print("    the saddle reading: a cell with dim E0 < q and |P_perp B2 v1| at round-off.")
NRAD = 2000


def stat_cell(tag, gen, n=12):
    rows = []
    for _ in range(n):
        s = setup(*gen())
        if s is None:
            continue
        v1, A0, B2, cf1, true, floor = s["v1"], s["A0"], s["B2"], s["cf1"], s["true"], s["floor"]
        g = B2 @ v1
        gp = g - v1 * (v1 @ g)
        gp = gp - v1 * (v1 @ gp)                    # TWICE: see the E3 note below
        nrm = float(np.linalg.norm(gp))
        # |gp| at round-off means v1 IS stationary for C, and normalising a noise vector AMPLIFIES its
        # leftover v1-component: v(t) = cos t v1 + sin t u then has |v| <> 1, B(v) > c^2 > ... > 1, and
        # the smaller root of F goes NEGATIVE (B(1-B) < 0).  My first E run reported 15793 'violations
        # of prop:chord' in the equal-angle cell for exactly this reason -- an artifact, not a result.
        stat = nrm < 1e-12 * float(np.linalg.norm(g))
        if stat:
            rows.append(dict(nrm=nrm, slope=0.0, pred=0.0, tmin=0.0, m0=s["m0"], dip=0.0,
                             viol=0, red=0.0, stat=True))
            continue
        B1 = float(v1 @ A0 @ v1)
        C1 = float(v1 @ B2 @ v1)
        Flam = abs(2.0 * C1 * cf1 - B1 - C1)
        pred = 2.0 * nrm * cf1 * (cf1 - 1.0) / Flam          # lam'(0+) along the descent sign

        def lam_of(t, sgn):
            v = np.cos(t) * v1 + sgn * np.sin(t) * (gp / nrm)
            v = v / np.linalg.norm(v)                       # unit by construction; enforce it anyway
            B = float(v @ A0 @ v); C = float(v @ B2 @ v)
            return float(crossings(np.array([B]), np.array([C]))[0])

        l0 = lam_of(0.0, 1.0)
        sl = {sg: (lam_of(1e-6, sg) - l0) / 1e-6 for sg in (1.0, -1.0)}
        sg = min(sl, key=lambda k: sl[k])                    # the sign that DECREASES lam
        ts = np.logspace(-6.0, np.log10(0.5 * np.pi), NRAD)
        lams = np.array([lam_of(t, sg) for t in ts])
        j = int(np.argmin(lams))
        rows.append(dict(nrm=nrm, slope=sl[sg], pred=pred, tmin=float(ts[j]), m0=s["m0"], stat=False,
                         l0err=float(l0 - cf1),
                         dip=float(lams[j] - l0), viol=int(np.sum((lams < true) & (true - lams > floor))),
                         red=float(1.0 - (lams[j] - true) / (cf1 - true)) if cf1 > true + floor else np.nan))
    if not rows:
        print(f"   {tag:>28}: no measurable draw")
        return
    nst = sum(1 for r in rows if r["stat"])
    act = [r for r in rows if not r["stat"]]
    print(f"   {tag:>28}: n={len(rows):2d} dimE0={sorted({r['m0'] for r in rows})}"
          f"  stationary(v1 is a true local min)={nst}  saddle={len(act)}"
          f"  lam'(0+)<0 in {sum(1 for r in act if r['slope'] < 0)}/{len(act)} saddles")
    if not act:
        print(f"{'':>32}  |P_perp B2 v1| max={max(r['nrm'] for r in rows):.3e} (round-off => no descent"
              f" direction exists; [A]'s 0 competitors is the matching observation)")
        return
    nm_ = np.array([r["nrm"] for r in act]);  dp_ = np.array([r["dip"] for r in act])
    sl_ = np.array([r["slope"] for r in act]); pr_ = np.array([r["pred"] for r in act])
    rd_ = np.array([r["red"] for r in act]);  tm_ = np.array([r["tmin"] for r in act])
    print(f"{'':>32}  |P_perp B2 v1|: min={nm_.min():.3e} max={nm_.max():.3e}   ray origin"
          f" |lam(0)-cf1| max={max(abs(r['l0err']) for r in act):.2e}"
          f"   viol(prop:chord) on rays={sum(r['viol'] for r in rows)}")
    print(f"{'':>32}  lam'(0+) vs prediction: max rel err={np.max(np.abs(sl_ - pr_) / np.abs(pr_)):.2e}"
          f"  slope med={np.median(sl_):.3e} pred med={np.median(pr_):.3e}")
    print(f"{'':>32}  dip along steepest ray: med={np.median(dp_):.3e} (#<0 {int(np.sum(dp_ < 0))}/{len(dp_)})"
          f"  t* med={np.median(tm_):.3e}  frac of cf1 gap removed med="
          f"{np.median(rd_):.3f} max={np.max(rd_):.3f}")


print("   [E1] generic cells where [A] found competitors:")
for nm, gen in [("q=3 spread 5-75", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 5-75"])))(lam_spd(6))),
                ("q=3 spread 30-88-89", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 30-88-89"])))(lam_spd(6))),
                ("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5))),
                ("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))]:
    stat_cell(nm, gen)
print("   [E2] generic cells where [A] found NONE (must still be saddles, just shallower than floor):")
for nm, gen in [("q=3 spread 2-45-88", lambda: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))(lam_spd(6))),
                ("q=3 2-45-88 mix", mixgen(SPECTRA["spread 2-45-88"], 3e-2))]:
    stat_cell(nm, gen)
print("   [E3] degenerate control (dim E0 = q): slope must be 0 and no competitor:")
stat_cell("q=3 equal 20d", lambda: (lambda L: (L, blocks(L, SPECTRA["equal 20d"])))(lam_spd(6)))

print("\n" + "=" * 118)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  A 每格 min_v lam - cf1 的最大值应在地板内（>=0 且 t* ~ 0，B(v*)/c^2 ~ 1）=> cf1 即家族最小元。")
print("  B 若某格出现 min_v lam < cf1 - 地板 => 存在更紧的可证单侧代理（家族最小值），报比值与位置。")
print("  C 任何 viol>0 => prop:chord 被推翻，正文必须重开，本文件其余内容无意义。")
print("  D ZK=0 控制组：cf1 = lam_min，故家族最小值应紧贴 cf1（差在地板），这是搜索本身的对照。")
print("  机制预登记（改完 v1 之后、跑之前写的）：dlam/dC<0、dlam/dB 在 B>(1-lam)/2 时也 <0，")
print("     而 t1 只最大化 E0 内的 C，所以 off-E0 方向可用 B 换 C；预言每个竞争者 Cratio>1 且 Bratio<1。")
print("     证伪：出现 Cratio<=1 的竞争者（脚本会单行打印 FALSIFIER）=> 机制解释错。")
print("  v1 自证 max|d0| 必须在地板内：首版我把 v1 取成 E0[:,-1]，退化 E0 下不是 C 的最大方向，")
print("     自证抓到 1.38e-02 的偏差，那一格的'竞争者'是我自己的错方向，不是真结果。")
print("  E 鞍点判据（跑前写的）：t1 只在 E0 内最大化 C，所以 |P_perp B2 v1| 一般非零 => v1 不是驻点，")
print("     lam 沿最陡下降射线一阶下降；预测 slope 与公式 lam'(0+)=C'(0)lam(lam-1)/|F_lam| 一致，")
print("     且射线上 viol 仍为 0（prop:chord 不破）。退化格（dim E0=q）斜率应为 0、无竞争者。")
print("     证伪：dim E0<q 而 |P_perp B2 v1| 落在舍入误差；或 slope 与公式不符；或射线上出现 viol。")
print("  已知局限（先写）：q>=4 的垂直球用的是 240 个随机点的网，不是覆盖网；t 网格 96 点。")
