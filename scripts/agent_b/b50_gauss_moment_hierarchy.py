"""
b50 -- the moment hierarchy behind prop:chord.  Motivation came from the literature leg of the b49
round (community.md 45.8): s_v(lambda) = B(v)^2/(B(v)-lambda C(v)) is NOT a new inequality, it is the
ONE-POINT Gauss rule for the Stieltjes transform of the positive measure
    rho_v = sum_i (a_i/ell_i) delta_{1/ell_i},   a_i = (k_i.v)^2,   Lambda = V diag(ell) V^T,
whose mass is B(v) and whose first moment is C(v).  The classical Gauss error formula says that for
h in C^{2n}
    E_n(h) = int h drho - G_n(h) = h^{(2n)}(xi)/(2n)! * ||p_n||^2  >= 0
with p_n the n-th monic orthogonal polynomial.  For h(y) = (1-lambda y)^{-1} every even derivative is
POSITIVE where lambda y < 1, so E_n >= 0 at ANY n, not only n=1.  Then verbatim the proof of
prop:chord gives, for every n and every unit v,

    lambda_n(v) := the smaller crossing of  G_n^v(lambda) = 1 - lambda    satisfies    lam_min(G) <= lambda_n(v),

and a better lower bound on phi_v means a crossing further DOWN, so this is a hierarchy of certified
upper bounds on lam_min(G) costing only more moments of Lambda (m_k = v^T K^T Lambda^{-(k+1)} K v) plus
one n x n symmetric tridiagonal eigenvalue problem.  If that works it DOMINATES the direction
optimisation of b48/b49 (a ~10% marginal on a 1e-7 quantity) and the note gets a sequence, not a formula.

IMPLEMENTATION NOTES (restated so a reader can check the algebra rather than trust the numbers):
  m_k = int y^k d rho_v = sum_r g_r^2 ell_r^{-(k+1)} with g = V^T K v, so m_0 = B(v), m_1 = C(v).
  Stieltjes procedure: p_{-1}=0, p_0=1 (monic), h_j = int p_j^2 d rho,
       a_j = int y p_j^2 d rho / h_j,   b_j = h_j / h_{j-1},   p_{j+1} = (y - a_j) p_j - b_j p_{j-1}.
  J_n = tridiag(a_0..a_{n-1}; sqrt(b_1)..sqrt(b_{n-1}));  nodes x_j = eig_j(J_n),
  weights w_j = m_0 * (first component of the j-th normalised eigenvector)^2;  G_n(l) = sum w_j/(1-l x_j).
  n=1: J_1 = [m_1/m_0], G_1 = m_0/(1 - l m_1/m_0) = B^2/(B - l C) = s_v, so lambda_1(v_1) must reproduce
  cf1 to round-off.  That is the self-check; if it fails nothing else in this file means anything.

PRE-REGISTERED before running (falsifiers stated as numbers, not as hopes):
  A  SIDE.  lambda_2, lambda_3, the exact crossing of phi_{v1} and lambda* must all sit >= lam_min(G)
     up to the per-draw floor.  A rule with n>=2 below lam_min(G) kills my reading of the Gauss error
     formula (most likely because the root sits above lambda_min(Lambda), where h leaves C^{2n} -- the
     very case split prop:chord's proof makes), and the hierarchy is dead as a claim.
  B  SELF-CHECK.  max|lambda_1(v_1) - cf1| <= 1e-12.
  B' EXACTNESS, and it comes FIRST (after two mis-indexed Jacobi matrices in this file, see
     gauss_rule): for every rule, sum_j w_j x_j^k must equal m_k to <= 1e-10 relative for k = 0..2n-1.
     A rule that integrates no polynomial exactly is not a Gauss rule, so a SIDE violation under it
     would be my bookkeeping and would say nothing about the error formula.  Plus one hand-computable
     two-atom measure printed before any random draw.
  C  TIGHTENING.  lambda_1 >= lambda_2 >= lambda_3, and the pointwise ordering G_1 <= G_2 <= G_3 <= phi
     at three test lambdas.  Exploratory, NOT a prediction: the error formula gives each rule being a
     lower bound, it does not give monotonicity between successive n.
  D  DOMINANCE -- the route question, fixed before running.  With gap = cf1 - lam_min(G), the moment
     step must remove a LARGER fraction of the gap than the whole direction optimisation:
        median (cf1 - lambda_2)/gap   vs   median (cf1 - cf2)/gap      (cf2 = b49's steepest-ray member)
     PREDICTION: n=2 removes ~0.3-1.0 of the gap, the ray removes 0.0004-0.50.
     FALSIFIER of "worth investing": median (cf1-lambda_2)/gap < median (cf1-cf2)/gap, in which case the
     hierarchy is decoration and I stop on it, keeping b49's formula as the last word on cf1.
  E  COST AND DEGENERACY.  breakdowns (h_j <= 0, i.e. rho_v carries fewer than j+1 atoms) and roots
     above lambda_min(Lambda) (the capped branch, where prop:chord's validity comes from the OTHER
     case split) must be counted, not silently dropped.
  F  ADDED AFTER THE FIRST RUN (the [A]-[E] block ran clean but compared everything to cf1, which is
     NOT the published bound -- prop:chord reads lam_min <= cf <= cf1, so a quantity in [cf, cf1]
     tightens nothing).  New, and written before this second run:  (cf - lambda_2)/(cf - lam_min) must
     be POSITIVE on a majority of draws, i.e. the moment step must sit BELOW the published cf.
     FALSIFIER: that fraction <= 0 everywhere => the hierarchy is a restatement of cf1 and I stop,
     keeping only the STRUCTURE (certified for every n, terminating at the number of atoms of rho_v).
  G  The n=2 and n=3 medians printed identically to four decimals against exact(v1) in run 1.  Either
     the moment truncation is genuinely negligible next to the FIXED-DIRECTION error (which would move
     the whole question back to maximising over v), or two of my rules are the same rule by accident.
     Measure |lambda_n - exact(v1)|/gap and count the atoms of rho_{v1} so the two are distinguishable.
"""
import numpy as np

rng = np.random.default_rng(50799)
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


def setup(Lam, K):
    """identical construction to b48/b49, plus the eigensystem of Lambda the moments are read off."""
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = float(np.linalg.eigvalsh(Gr)[0])
    ev, V = np.linalg.eigh(Lam)
    A0 = K.T @ np.linalg.solve(Lam, K)
    B2 = K.T @ np.linalg.solve(Lam, np.linalg.solve(Lam, K))
    w0, V0 = np.linalg.eigh(A0)
    c2 = float(w0[-1])
    if c2 > 1.0 + 1e-12 or c2 <= 0.0:
        return None
    tol = 1e-9 * max(abs(c2), 1.0)
    E0 = V0[:, w0 >= w0[-1] - tol]
    mu, W = np.linalg.eigh(E0.T @ B2 @ E0)
    t1 = float(mu[-1]) / c2
    cf1 = cf_of(t1, c2)
    v1 = E0 @ W[:, -1]
    v1 = v1 / np.linalg.norm(v1)
    t_global = float(np.linalg.eigvalsh(B2)[-1]) / c2
    return dict(true=true, A0=A0, B2=B2, c2=c2, t1=t1, cf1=cf1, v1=v1, m0=E0.shape[1],
                cf=cf_of(t_global, c2), t_global=t_global,
                V=V, ell=ev, Kv=K @ v1, lminL=float(ev[0]), floor=100.0 * EPS * np.linalg.norm(Gr, 2))


def moments(s, order):
    g = s["V"].T @ s["Kv"]
    w = g * g
    return np.array([float(np.sum(w * s["ell"] ** -(k + 1))) for k in range(order + 1)])


def gauss_rule(m, n):
    """nodes/weights of the n-point Gauss rule for the measure with moments m[0..2n]; None on
    breakdown (the measure carries fewer than n atoms, in which case a smaller n IS exact)."""
    a = np.zeros(n)
    b = np.zeros(n)
    h_prev, h_cur = 1.0, m[0]
    pjm1 = np.zeros(n + 2)
    pj = np.zeros(n + 2)
    pj[0] = 1.0                                    # p_0 = 1, monic, ascending coefficients
    for j in range(n):
        aj = 0.0
        for i in range(j + 1):
            for k in range(j + 1):
                aj += pj[i] * pj[k] * m[i + k + 1]
        a[j] = aj / h_cur
        bj = h_cur / h_prev                       # b_j = h_j/h_{j-1}, the off-diagonal at slot j.
        b[j] = bj                                 # MUST be stored BEFORE the break: the last
        # coefficient the Jacobi matrix needs is b_{n-1} = h_{n-1}/h_{n-2}, which is produced by the
        # final iteration (j = n-1) even though that iteration builds no polynomial.  Both of my
        # earlier attempts mis-set this (slot j+1, then a slot that stayed zero for j = n-1) and both
        # produced a rule that is not a Gauss rule -- see exactness().
        if j == n - 1:
            break
        pn = np.zeros(n + 2)
        pn[1:j + 2] += pj[:j + 1]                 # y p_j
        pn[:j + 1] -= a[j] * pj[:j + 1]           # - a_j p_j
        if j > 0:
            pn[:j] -= bj * pjm1[:j]               # - b_j p_{j-1}
        hn = 0.0
        for i in range(j + 2):
            for k in range(j + 2):
                hn += pn[i] * pn[k] * m[i + k]
        if not (hn > 0.0) or not np.isfinite(hn):
            return None
        pjm1, pj, h_prev, h_cur = pj, pn, h_cur, hn
    J = np.diag(a)
    for j in range(1, n):
        J[j - 1, j] = J[j, j - 1] = np.sqrt(b[j])
    ev, Vv = np.linalg.eigh(J)
    return ev, m[0] * Vv[0, :] ** 2


def exactness(x, w, m, n):
    """max_k |sum_j w_j x_j^k - m_k| / |m_k| over k = 0..2n-1.  This is the check that has to come
    BEFORE the side test: a quadrature rule that does not integrate polynomials of degree < 2n exactly
    is not a Gauss rule, so its 'lower bound' property is mine, not the theorem's, and any SIDE
    violation would then say nothing about prop:chord."""
    err = 0.0
    for k in range(2 * n):
        approx = float(np.sum(w * x ** k))
        denom = abs(m[k]) if abs(m[k]) > 0 else 1.0
        err = max(err, abs(approx - m[k]) / denom)
    return err


def cubic_two(mm):
    """lambda_2 as the smallest non-negative root of an explicit CUBIC in m0..m3 (derivation in the
    [H] comment of cell()); no eigenproblem, no quadrature nodes."""
    m0, m1, m2, m3 = mm[0], mm[1], mm[2], mm[3]
    a0 = m1 / m0
    b1 = m2 / m0 - a0 * a0
    if not (b1 > 0.0):
        return np.nan                                     # rho carries < 2 atoms: n=2 is the exact rule
    a1 = (m3 - 2.0 * a0 * m2 + a0 * a0 * m1) / (m0 * b1)
    s1, p2 = a0 + a1, a0 * a1 - b1
    rts = np.roots([p2, -(p2 + s1), (s1 + 1.0) - (m0 * s1 - m1), m0 - 1.0])
    real = np.array([z.real for z in rts if abs(z.imag) < 1e-9 and z.real >= 0.0])
    if real.size == 0:
        return np.nan
    lo = float(real.min())
    def g(l):
        return (m0 - l * (m0 * s1 - m1)) / (1.0 - s1 * l + p2 * l * l) - (1.0 - l)
    if abs(g(lo)) > 1e-6 * max(1.0, m0):
        return np.nan                                     # not a root of G_2 = 1-l after the
    # (1-l)D multiplication went through a zero of D -- [H] measures the agreement either way
    return lo


def bisect_cross(gfun, hi, it=200):
    """root of an increasing-convex-minus-decreasing function on [0, hi); NaN when the sign never
    changes inside the bracket (then the crossing is above the rule's pole, reported as such)."""
    if gfun(hi) < 0.0:
        return np.nan
    lo = 0.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if gfun(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def rule_cross(x, w):
    xmax = float(np.max(x))
    pole = 1.0 / xmax * (1.0 - 1e-12)
    return bisect_cross(lambda l: float(np.sum(w / (1.0 - l * x))) - (1.0 - l), pole)


def phi_cross(s):
    g = s["V"].T @ s["Kv"]
    w, ell = g * g, s["ell"]
    top = float(ell[0]) * (1.0 - 1e-12)
    return bisect_cross(lambda l: float(np.sum(w / (ell - l))) - (1.0 - l), top)


def fstar_cross(s, K, Lam):
    top = s["lminL"] * (1.0 - 1e-12)
    I = np.eye(Lam.shape[0])
    def gg(l):
        M = K.T @ np.linalg.solve(Lam - l * I, K)
        return float(np.linalg.eigvalsh(M)[-1]) - (1.0 - l)
    return bisect_cross(gg, top, it=120)


def ray_cf2(s):
    """b49's steepest-ray member evaluated EXACTLY at t* = -L1/(2 L2) (not to second order)."""
    v1, A0, B2, lam, c2 = s["v1"], s["A0"], s["B2"], s["cf1"], s["c2"]
    g = B2 @ v1
    gp = g - v1 * (v1 @ g)
    gp = gp - v1 * (v1 @ gp)
    nrm = float(np.linalg.norm(gp))
    if nrm < 1e-12 * float(np.linalg.norm(g)):
        return np.nan, True                                    # stationary: no ray, no cf2
    u = gp / nrm
    B0, C0 = c2, float(v1 @ B2 @ v1)
    b2 = float(u @ A0 @ u) - B0
    c1, c2t = 2.0 * nrm, float(u @ B2 @ u) - C0
    Fl = 2.0 * C0 * lam - B0 - C0
    Fll, FB, FC, FlC = 2.0 * C0, 1.0 - lam - 2.0 * B0, lam * lam - lam, 2.0 * lam - 1.0
    L1 = -FC * c1 / Fl
    L2 = -(Fll * L1 * L1 + 2.0 * FlC * L1 * c1 + 2.0 * FB * b2 + 2.0 * FC * c2t) / (2.0 * Fl)
    if not (L2 > 0.0):
        return np.nan, False
    tp = -L1 / (2.0 * L2)
    v = np.cos(tp) * v1 + np.sin(tp) * u
    v = v / np.linalg.norm(v)
    return float(crossings(np.array([float(v @ A0 @ v)]), np.array([float(v @ B2 @ v)]))[0]), False


def cell(tag, gen, n=N):
    rows = []
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        m = moments(s, 6)
        r = dict(m0=s["m0"], cf1=s["cf1"], true=s["true"], floor=s["floor"],
                 gap=s["cf1"] - s["true"], lminL=s["lminL"], cf=s["cf"],
                 tgap=s["cf"] - s["true"])
        aw = (s["V"].T @ s["Kv"]) ** 2 / s["ell"]
        keep = aw > 1e-13 * aw.max()
        r["natom"] = int(len(np.unique(np.round(1.0 / s["ell"][keep], 12))))
        g1, g2, g3 = gauss_rule(m, 1), gauss_rule(m, 2), gauss_rule(m, 3)
        r["ex1"] = exactness(*g1, m, 1) if g1 is not None else np.nan
        r["ex2"] = exactness(*g2, m, 2) if g2 is not None else np.nan
        r["ex3"] = exactness(*g3, m, 3) if g3 is not None else np.nan
        r["mom"] = m
        r["l1"] = rule_cross(*g1) if g1 is not None else np.nan
        r["l2"] = rule_cross(*g2) if g2 is not None else np.nan
        r["l3"] = rule_cross(*g3) if g3 is not None else np.nan
        r["bd2"] = g2 is None
        r["bd3"] = g3 is None
        r["lexact"] = phi_cross(s)
        r["lstar"] = fstar_cross(s, K, Lam)
        r["cf2"], r["stat"] = ray_cf2(s)
        # pointwise ordering G_1 <= G_2 <= G_3 <= phi, at lambdas strictly below lambda_min(Lambda)
        nok = nn = 0
        for frac in (0.3, 0.6, 0.9):
            l = frac * s["cf1"]
            if l < s["lminL"] and g1 and g2:
                vals = [float(np.sum(w / (1.0 - l * x))) for x, w in (g1, g2)]
                if g3 is not None:
                    vals.append(float(np.sum(g3[1] / (1.0 - l * g3[0]))))
                gg = s["V"].T @ s["Kv"]
                vals.append(float(np.sum(gg * gg / (s["ell"] - l))))
                nn += 1
                nok += int(all(vals[i] <= vals[i + 1] + 1e-13 for i in range(len(vals) - 1)))
        r["nok"], r["nn"] = nok, nn
        rows.append(r)

    act = [r for r in rows if not r["stat"]]
    st = [r for r in rows if r["stat"]]
    if not act:
        print(f"   {tag:>26}: n={len(rows):3d} dimE0={sorted({r['m0'] for r in rows})}"
              f" stationary={len(st)}  (no ray and no hierarchy test in this cell)")
        return
    def viol(key):
        return sum(int(r[key] == r[key] and r[key] < r["true"] - r["floor"]) for r in act)
    def frac(key):
        v = np.array([(r["cf1"] - r[key]) / r["gap"]
                      if r[key] == r[key] and r["gap"] > r["floor"] else np.nan for r in act])
        return v
    sc = np.array([abs(r["l1"] - r["cf1"]) for r in act if r["l1"] == r["l1"]])
    cap2 = sum(int(r["l2"] == r["l2"] and r["l2"] > r["lminL"]) for r in act)
    cap3 = sum(int(r["l3"] == r["l3"] and r["l3"] > r["lminL"]) for r in act)
    n12 = sum(int(r["l1"] == r["l1"] and r["l2"] == r["l2"] and r["l1"] >= r["l2"] - r["floor"]) for r in act)
    n23 = sum(int(r["l2"] == r["l2"] and r["l3"] == r["l3"] and r["l2"] >= r["l3"] - r["floor"]) for r in act)
    print(f"   {tag:>26}: n={len(rows):3d} dimE0={sorted({r['m0'] for r in rows})}"
          f" stationary={len(st)}  [A] SIDE viol n=2:{viol('l2')} n=3:{viol('l3')}"
          f" exact:{viol('lexact')} lam*:{viol('lstar')} cf2:{viol('cf2')}   (all must be 0)"
          f"   [E] breakdown n=2:{sum(r['bd2'] for r in act)} n=3:{sum(r['bd3'] for r in act)}"
          f" capped n=2:{cap2} n=3:{cap3}")
    def mx(key):
        v = np.array([r[key] for r in act])
        return np.nanmax(v) if np.isfinite(v).any() else float("nan")

    print(f"{'':>28}  [B'] moment exactness max_k rel err: n=1 {mx('ex1'):.1e}  n=2 {mx('ex2'):.1e}"
          f"  n=3 {mx('ex3'):.1e}   (must be <= 1e-10; if not, these are NOT Gauss rules and"
          f" the [A] violations are my bookkeeping, not the theorem)")
    print(f"{'':>28}  [B] max|lambda_1-cf1| = {sc.max() if sc.size else float('nan'):.1e}"
          f"   [C] l1>=l2 {n12}/{len(act)}, l2>=l3 {n23}/{len(act)},"
          f" pointwise G-ordering {sum(r['nok'] for r in act)}/{sum(r['nn'] for r in act)}")
    for key, lab in (("l2", "n=2"), ("l3", "n=3"), ("lexact", "exact(v1)"), ("cf2", "b49 ray"),
                     ("lstar", "lam*")):
        v = frac(key)
        if np.isfinite(v).any():
            print(f"{'':>28}   [{lab:>10}] of cf1's gap removed: med={np.nanmedian(v):7.4f}"
                  f" min={np.nanmin(v):7.4f} max={np.nanmax(v):7.4f}  n={int(np.isfinite(v).sum())}")
    # ---- [F]: ADDED AFTER THE FIRST RUN, then RE-POSED AFTER READING eq:chain1 -- and the re-posing
    # is the finding, not a bookkeeping detail.  My premise was "cf is the published bound, so the new
    # quantity must beat cf".  prop:chord claims only  lam_min <= cf1  and  cf <= cf1:  in the no-angle
    # model cf may sit BELOW lam_min, which is exactly the content of prop:side's D = f - g and of
    # retraction (m).  The run confirmed it on every cell (med(cf-lam_min) < 0 while med(cf1-lam_min)
    # > 0 on the SAME draws), so [F] is a diagnostic of which bound is live here, and cf1 is the target.
    cfp = sum(int(r["cf"] <= r["cf1"] + r["floor"]) for r in act)
    cfok = sum(int(r["cf"] >= r["true"] - r["floor"]) for r in act)
    print(f"{'':>28}   [F] cf <= cf1 in {cfp}/{len(act)} (eq:chain1, re-checked).  cf >= lam_min in"
          f" {cfok}/{len(act)} -- as expected cf is NOT valid without the angle hypothesis here,"
          f" so the live target is cf1, not cf.")
    print(f"{'':>28}       med(cf-lam_min) = {np.median([r['tgap'] for r in act]):+.3e},"
          f" med(cf1-lam_min) = {np.median([r['gap'] for r in act]):+.3e},"
          f" med(cf1-cf) = {np.median([r['cf1'] - r['cf'] for r in act]):+.3e}")
    ag2 = [abs(r["l2"] - r["lexact"]) / r["gap"] for r in act
           if r["l2"] == r["l2"] and r["lexact"] == r["lexact"] and r["gap"] > r["floor"]]
    ag3 = [abs(r["l3"] - r["lexact"]) / r["gap"] for r in act
           if r["l3"] == r["l3"] and r["lexact"] == r["lexact"] and r["gap"] > r["floor"]]
    print(f"{'':>28}   [G] max|lambda_n - exact(v1)|/gap: n=2 {max(ag2) if ag2 else float('nan'):.2e}"
          f"  n=3 {max(ag3) if ag3 else float('nan'):.2e}"
          f"   (atoms of rho_v1 with weight > 1e-13: {sorted({r['natom'] for r in act})},"
          f" rule uses n=2 -> the moment truncation is NOT the bottleneck; the fixed direction is)")
    # ---- [H] THE n=2 RULE AS AN EXPLICIT CUBIC IN FOUR SCALARS (this is what makes the bound a
    # formula rather than an algorithm).  With a0=m1/m0, b1=m2/m0-a0^2, h1=m0*b1,
    # a1=(m3-2a0 m2+a0^2 m1)/h1, s1=a0+a1, p2=a0 a1-b1:
    #   G_2(l) = [m0 - l(m0 s1 - m1)] / [1 - s1 l + p2 l^2],
    #   G_2(l) = 1-l  <=>  p2 l^3 - (p2+s1) l^2 + [(s1+1) - (m0 s1 - m1)] l + (m0-1) = 0.
    # FALSIFIER: any disagreement with the eigen/bisection lambda_2 above => my reduction of
    # nodes/weights to moments is wrong (I derived w1 x2 + w2 x1 = m0 s1 - m1 by hand, memory rule 5).
    dc, ncf = [], 0
    for r in act:
        if r["l2"] != r["l2"]:
            continue
        rt = cubic_two(r["mom"])
        if rt == rt:
            ncf += 1
            dc.append(abs(rt - r["l2"]))
    print(f"{'':>28}   [H] explicit-cubic lambda_2 vs eigen+bisection lambda_2:"
          f" max abs diff = {max(dc) if dc else float('nan'):.2e} over {ncf}/{len(act)} draws")


def mixgen(cs, delta):
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


print("=" * 124)
print("[B'] HAND-COMPUTABLE CASE, run before any random draw is trusted.")
print("    rho = 0.2*delta_{1/3} + 0.3*delta_{1}:  m_k = 0.2*3^-k + 0.3; m_0=0.5, m_1=11/30, m_2=29/90.")
print("    (MASS 0.5 <= 1 ON PURPOSE: the crossing G(0)=m_0 must sit BELOW 1-0, else the smaller root")
print("    is negative and my bisection on [0,1) reports 0 -- the first version of this toy had m_0=3")
print("    and printed three meaningless zeros.)")
print("    1-point rule: node m1/m0 = 11/15 = 0.733333..., weight m_0 = 0.5 (exact through degree 1;")
print("    G_1(y^2)=0.5*(11/15)^2=0.268889 < m_2=0.322222, i.e. strictly below phi, as E_1 >= 0 says).")
print("    2-point rule for a 2-atom measure must reproduce the measure: nodes {1/3,1}, weights")
print("    {0.2,0.3}, exactness ~0, and then lambda_2 = the EXACT crossing -- the hierarchy terminates")
print("    at n = (number of atoms), which is the honest statement of what 'every n' buys.")
mm = np.array([0.2 * 3.0 ** (-k) + 0.3 for k in range(7)])
for nn in (1, 2):
    gr = gauss_rule(mm, nn)
    print(f"    n={nn}: nodes {np.array2string(gr[0], precision=12)}"
          f"  weights {np.array2string(gr[1], precision=12)}"
          f"  exactness {exactness(*gr, mm, nn):.3e}"
          f"  cross {rule_cross(*gr):.12f}")
print(f"    exact phi crossing on this measure (root of 0.2/(1-l/3)+0.3/(1-l)=1-l): "
      f"{bisect_cross(lambda l: 0.2/(1.0-l/3.0)+0.3/(1.0-l)-(1.0-l), 1.0-1e-12):.12f}")
print(f"    explicit cubic [H] on the same toy measure: lambda_2 = {cubic_two(mm):.12f}"
      f"  (must equal the line above; this is the hand case for the closed form)")

print("\n" + "=" * 124)
print("[A-E] Gauss-moment hierarchy over prop:chord: lambda_n(v1) for n=1,2,3 points of rho_{v1},")
print("      against b49's one-step direction fix (cf2), the exact crossing of phi_{v1} and lam*.")
for name, cs in SPECTRA.items():
    cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    cell(nm, mixgen(cs, delta))
cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 124)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  A 侧面：lambda_2/lambda_3/exact(v1)/lam* 都不得落在 lam_min(G) 以下（超出 floor 即违反）。")
print("     违反 => 我对 Gauss 误差公式的使用错（根可能在 lam_min(Lambda) 上方，h 离开 C^{2n}），")
print("     层级作废，只保留 b49 的射线。")
print("  B 自证：max|lambda_1 - cf1| <= 1e-12，否则测度/Jacobi 归一化有错。")
print("  C 收紧：l1>=l2>=l3 的比例；逐点 G_1<=G_2<=G_3<=phi 的成立数。误差公式只保证'每个都偏下'，")
print("     不保证 n 之间单调，所以这条是探索而非预言。")
print("  D 支配性（路线判据，跑前定死）：med (cf1-lambda_2)/gap 必须 > med (cf1-cf2)/gap，")
print("     否则层级是装饰，停止投入。预言 n=2 拿走 0.3..1.0。")
print("  E 代价与退化：breakdown（原子数 < n）与根越过 lam_min(Lambda) 的 capped 计数。")
print("  F 首跑后新增、读章节后重设前提：prop:chord 只声明 lam_min<=cf1 与 cf<=cf1；无角度假设时 cf")
print("     可以低于 lam_min（这正是 prop:side 的 D=f-g 与撤回(m) 的内容）。故 cf 不是被比较的界，")
print("     cf1 才是。实测每格 med(cf-lam_min)<0 而 med(cf1-lam_min)>0，与章节一致，不是反例。")
print("  G |lambda_n - exact(v1)|/gap 与 rho_v1 的原子数：判'瓶颈在矩截断还是在固定方向'。")
print("  H n=2 的界写成 m0..m3 的显式三次方程最小根 —— 与特征值+二分版本必须一致（<=1e-12），")
print("     否则我手推的 w1x2+w2x1 = m0*s1 - m1 归约错，'闭式'一句作废。")
