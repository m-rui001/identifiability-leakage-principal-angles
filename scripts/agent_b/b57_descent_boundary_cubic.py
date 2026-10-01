"""
b57 -- the boundary of the n=1 descent set has a CLOSED FORM: a cubic in tan(theta) whose coefficients are
six scalars, and whose leading root is 2g/P.  Derived on paper before running, then tested as an IDENTITY
(not a fit).

STARTING POINT.  b52 measured that {v : lambda_1(v) < cf_1} is a boundary layer of angular width
t_hi = 2.5e-5..1.6e-3 around v_1, and b55 turned the ordering into the exact criterion
    lambda_1(v) <= cf_1  <=>  T(v) >= 0,   T := t - t_1 - kappa*(c^2 - B),   kappa := (1 - t_1 mu)/(c^2 mu),
with mu = cf_1.  Those two facts are the SAME fact: the boundary of the descent set is exactly {T = 0},
with no truncation anywhere, because the criterion is an iff.  So the layer's width is COMPUTABLE.
Take a ray v(theta) = cos(theta) v_1 + sin(theta) u, u unit, u perp v_1.  Then

    B(theta) = c^2 - D sin^2(theta),        D := c^2 - u'A_0u >= 0,
        [the cross term vanishes for EVERY u perp v_1: v_1 is a top eigenvector of A_0, so
         v_1'A_0u = c^2 (v_1.u) = 0 -- this is a structural fact about A_0's eigenvector, not a choice]
    C(theta) = t_1 c^2 + 2g sin cos + (b - t_1 c^2) sin^2,   g := v_1'B_2u,  b := u'B_2u,
    C - t_1 B = 2g sin cos + E sin^2,       E := b - t_1 a,   a := u'A_0u,
    T = (C - t_1 B)/B - kappa D sin^2.

Put x = tan(theta) (>0 branch) and divide by sin; then with  P := kappa D c^2 - E,
    (P - kappa D^2) x^3 - 2g x^2 + P x - 2g = 0                      (eq:cubic, claimed)
is EXACTLY the descent-set boundary along the ray, and
    T'(0) = A_1 = 2g/c^2,   T''(0)/2 = A_2 = E/c^2 - kappa D = -P/c^2,
    leading root x_hi = 2g/P = A_1/|A_2|,  stationary angle g/P = x_hi/2,
    dip = |dr/dt| * A_1^2/(4|A_2|) = |dr/dt| * g^2/(c^2 P),   dr/dt = mu(1-mu)/(2 t_1 mu - 1 - t_1) < 0.
P > 0 <=> A_2 < 0 <=> the descent set along that ray is a BOUNDED layer; P <= 0 is the alternative
topology (unbounded layer), counted rather than averaged.

CROSS-LANE CHECK (the one I like most).  b49 computed L_1 = -F_C c_1/F_lambda on the same steepest ray.
With B_0 = c^2, C_0 = t_1 c^2:  F_lambda = 2C_0 mu - B_0 - C_0 = c^2(2 t_1 mu - 1 - t_1),
F_C = mu^2 - mu = -mu(1-mu), c_1 = 2g, so
    L_1(b49) = mu(1-mu) * 2g / [c^2 (2 t_1 mu - 1 - t_1)] = (dr/dt)*(2g/c^2) = (dr/dt)*A_1,
which is ALSO what dlambda_1/dtheta|_0 = (dr/dB)B'(0) + (dr/dt)t'(0) gives since B'(0)=0 and
t'(0) = C'(0)/c^2 = 2g/c^2.  Two independent derivations, one algebraic identity: the ratio is 1 to
round-off.  FALSIFIER: |ratio-1| > 1e-8 -- then either b49's L_1 or my coordinate change is wrong, and
the chapter's rem:chord dip formula and lem:margin cannot both stand.

PRE-REGISTERED, falsifiers as numbers.  Signs are registered ONLY where a remainder is bounded (rule 8);
magnitudes are printed as medians AND per-draw extremes, labelled (rule 13).
  [T0] GATE, ray identities on the steepest u and on 3 random u perp v_1:
       |v_1'A_0u|/c^2 <= 1e-12;  max|B-Bc|/c^2 and max|C-Cc|/|t_1c^2| <= 1e-12 over a 3000-point
       GEOMETRIC theta grid;  T from its definition vs T from the closed form <= 1e-12 relative;
       cubic residual at its own positive root <= 1e-12 (scaled by max|coef|).
       FALSIFIER: any > 1e-10 -- then the expansion is wrong, not the numerics, and later legs are void.
       (This is retraction (p)'s failure mode: a rewrite used because it looked familiar.)
  [T1] THE BOUNDARY IS EXACT.  Bisect the EXACT lambda_1(v(theta)) - cf_1 for its first sign change and
       compare with the cubic's smallest positive root: |theta_bisect/theta_cubic - 1| <= 1e-8 predicted,
       because BOTH are exact -- this is a second computation of one number, not a model test.
       FALSIFIER: > 1e-6 -- then the iff of lem:margin has a direction-wise exception, i.e. b55's lemma
       is wrong.  That is the most expensive outcome available, which is why the reference is an
       independent root finder and not my own algebra.
       x_hi = 2g/P vs the exact theta*, and g/P vs b49's tp, are TRUNCATIONS: reported, no sign.
  [T2] CROSS-LANE: L_1/[(dr/dt)A_1] - 1 <= 1e-8 (forced above).  FALSIFIER > 1e-6.
       L_2/[(dr/dt)A_2] and dip(b49) vs |dr/dt|g^2/(c^2P) differ by r's own second-order terms, so a
       ratio away from 1 is information about which truncation is cheaper, NOT a refutation: no sign.
  [T3] EQUAL-ANGLE ENDPOINT from the ray side: on equal-20d, D=0 for every u and g=0 (because v_1 IS the
       C-maximiser, so B_2v_1 = lam_max(B_2)v_1 and v_1'B_2u = lam_max(B_2)(v_1.u) = 0), hence the cubic
       collapses to  P x (x^2 + 1) = 0  with  P = t_1 c^2 - b > 0,  whose only real root is x = 0:
       the descent boundary sits AT v_1, i.e. the descent set is EMPTY.
       FALSIFIER: any theta with lambda_1(v(theta)) < cf_1 - floor.  Also, in a GENERIC cell the count of
       draws with g <= 1e-12 must be 0, else x_hi=0 holds trivially instead of structurally.
  [T4] THE INTERPOLATION QUESTION (§48.11(b)) -- and here my first instinct was WRONG, which is why this
       leg is worth running at all.  I first predicted x_hi ~ delta (the layer collapsing as the cosines
       equalise).  It does not, for a reason I can derive: in this generator A_0 = Q_2 diag(cos^2) Q_2^T,
       so v_1 = Q_2's column of the LARGEST cosine, which is INDEPENDENT of delta, while
       B_2 = Q_2 Dc (Q_1^T Lam^-1 Q_1) Dc Q_2^T is NOT diagonal in that basis.  Hence as delta -> 0+
         D = c^2 - a -> 0  at slope 1,   g -> g_0 = O(1),   P -> P_0 = -E_0 > 0 at slope 0,
         x_hi -> 2 g_0/P_0 = O(1)   (slope 0),  dip -> O(1) (slope 0).
       The equal-angle point is an ISOLATED point, not a limit: at delta = 0 exactly, dim E_0 jumps to q
       and v_1 is re-selected as the C-MAXIMISER over all of R^q, for which g = 0.  So the descent layer
       does NOT collapse continuously -- there is no interpolation statement in eta = c^2 - lam_min(A_0).
       REGISTERED (bounded remainder: the cubic's correction is O(x^2) relative, so the two-jet root is
       provably leading as delta -> 0):
         slope of log x_hi vs log delta in [-0.2, 0.2];   slope of log g in [-0.2, 0.2];
         slope of log D in [0.7, 1.3];                    slope of log P in [-0.2, 0.2];
         slope of log(dip/gap) in [-0.2, 0.2].
       FALSIFIER: slope of x_hi > 0.5 -- then the layer DOES collapse and §48.11(b) reopens as a
       continuous statement; either way the answer is the slope, so both branches are information (rule 6).
  [T5] MEASUREMENT, no prediction: angle(v*, v_1)/theta_bisect per cell.  b53 reported the two numbers
       equal in ORDER; if the ratio is not O(1) then §47.3's "second computation of the same quantity"
       was a coincidence of magnitude and I will say so.
  [T6] SIGNS COUNTED, not averaged: #(P<=0), #(L2<=0), #(they disagree), and #(draws whose bisected scan
       finds NO upper crossing) -- the last is the empirical face of P<=0.  B-CLAIM-56(i) wanted L_2>0
       proved; this leg reduces it to the scalar inequality kappa(c^2-a)c^2 > b - t_1 a at the steepest
       direction, which at q=2 is EXHAUSTIVE (one perpendicular direction, so P is a number per draw).
Draws: seed 60311 with b53/b55's cell order, so the main cells are the SAME 320 draws (setup/crossings/
ray helpers copied verbatim, q-block assert kept).  [T4]'s ladder switches to seed 51400 AFTER that
section and announces it.
"""
import numpy as np

rng = np.random.default_rng(60311)
N = 40
EPS = np.finfo(float).eps
TH = np.geomspace(1e-10, np.pi / 2 - 1e-12, 3000)      # geometric on purpose: b51's linspace ceiling
SC, SN = np.cos(TH), np.sin(TH)

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
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    w, W = np.linalg.eigh(Gr)
    true = float(w[0])
    y = W[p:, 0]
    assert len(y) == q and len(W[:p, 0]) == p
    vstar = y / np.linalg.norm(y)
    ev, V = np.linalg.eigh(Lam)
    Li = V @ np.diag(1.0 / ev) @ V.T
    Mm = [K.T @ Li @ K]
    P = Li
    for k in range(2, 5):
        P = P @ Li
        Mm.append(K.T @ P @ K)
    A0, B2 = Mm[0], Mm[1]
    w0, V0 = np.linalg.eigh(A0)
    c2 = float(w0[-1])
    if c2 > 1.0 + 1e-12 or c2 <= 0.0:
        return None
    tol = 1e-9 * max(abs(c2), 1.0)
    E0 = V0[:, w0 >= w0[-1] - tol]
    mu, Wm = np.linalg.eigh(E0.T @ B2 @ E0)
    t1 = float(mu[-1]) / c2
    cf1 = cf_of(t1, c2)
    v1 = E0 @ Wm[:, -1]
    v1 = v1 / np.linalg.norm(v1)
    if cf1 >= min(float(ev[0]), 1.0):
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, t1=t1, cf1=cf1, v1=v1, vstar=vstar,
                lam_minL=float(ev[0]), M=Mm, K=K, Lam=Lam,
                floor=100.0 * EPS * np.linalg.norm(Gr, 2), m0=E0.shape[1])


def ray_L(s):
    """b49/b50's steepest ray, VERBATIM from b53/b55, extended only to also return L1, L2, tp, dip."""
    v1, A0, B2, lam, c2 = s["v1"], s["A0"], s["B2"], s["cf1"], s["c2"]
    g = B2 @ v1
    gp = g - v1 * (v1 @ g)
    gp = gp - v1 * (v1 @ gp)
    nrm = float(np.linalg.norm(gp))
    if nrm < 1e-12 * float(np.linalg.norm(g)):
        return None
    u = gp / nrm
    B0, C0 = c2, float(v1 @ B2 @ v1)
    b2 = float(u @ A0 @ u) - B0
    c1, c2t = 2.0 * nrm, float(u @ B2 @ u) - C0
    Fl = 2.0 * C0 * lam - B0 - C0
    Fll, FB, FC, FlC = 2.0 * C0, 1.0 - lam - 2.0 * B0, lam * lam - lam, 2.0 * lam - 1.0
    L1 = -FC * c1 / Fl
    L2 = -(Fll * L1 * L1 + 2.0 * FlC * L1 * c1 + 2.0 * FB * b2 + 2.0 * FC * c2t) / (2.0 * Fl)
    if not (L2 > 0.0):
        return u, nrm, L1, L2, np.nan, np.nan
    tp = -L1 / (2.0 * L2)
    return u, nrm, L1, L2, tp, -L1 * L1 / (4.0 * L2)


def scalars(s, u):
    v1, A0, B2, c2, t1, mu = s["v1"], s["A0"], s["B2"], s["c2"], s["t1"], s["cf1"]
    a = float(u @ A0 @ u)
    b = float(u @ B2 @ u)
    g = float(v1 @ B2 @ u)
    cross = float(v1 @ A0 @ u)
    D = c2 - a
    E = b - t1 * a
    kap = (1.0 - t1 * mu) / (c2 * mu)
    P = kap * D * c2 - E
    coef = np.array([P - kap * D * D, -2.0 * g, P, -2.0 * g])
    drdt = mu * (1.0 - mu) / (2.0 * t1 * mu - 1.0 - t1)
    return dict(a=a, b=b, g=g, cross=cross, D=D, E=E, kap=kap, P=P, coef=coef,
                drdt=drdt, A1=2.0 * g / c2, A2=E / c2 - kap * D)


def lam1_at(A0, B2, v1, u, th):
    """lambda_1 of the family at the ray point cos(th) v1 + sin(th) u, by the independent solver."""
    vv = np.cos(th) * v1 + np.sin(th) * u
    return float(crossings(np.array([float(vv @ A0 @ vv)]), np.array([float(vv @ B2 @ vv)]))[0])


def ray_check(s, u):
    """all per-ray numbers: identity errors, cubic roots, exact bisected boundary, competitor count."""
    c2, t1, cf1, A0, B2, v1 = s["c2"], s["t1"], s["cf1"], s["A0"], s["B2"], s["v1"]
    sc = scalars(s, u)
    V = np.outer(SC, v1) + np.outer(SN, u)
    Bv = np.einsum("ij,jk,ik->i", V, A0, V)
    Cv = np.einsum("ij,jk,ik->i", V, B2, V)
    Bc = c2 - sc["D"] * SN ** 2
    Cc = t1 * c2 + 2.0 * sc["g"] * SC * SN + (sc["b"] - t1 * c2) * SN ** 2
    Tdef = Cv / Bv - t1 - sc["kap"] * (c2 - Bv)
    Tcl = (Cc - t1 * Bc) / Bc - sc["kap"] * sc["D"] * SN ** 2
    d = dict(sc)
    d["eB"] = float(np.max(np.abs(Bv - Bc)) / c2)
    d["eC"] = float(np.max(np.abs(Cv - Cc)) / max(abs(t1 * c2), 1e-30))
    d["eT"] = float(np.max(np.abs(Tdef - Tcl)) / max(1e-30, np.max(np.abs(Tcl))))
    d["crossrel"] = abs(sc["cross"]) / c2
    lam = crossings(Bv, Cv)
    gapv = max(cf1 - s["true"], 1e-30)
    thr = 1.0e-12 * gapv            # DESCENT PREDICATE, not bare f<0: see the defect note below
    d["below"] = int(np.sum(lam < cf1 - s["floor"]))
    d["nDesc"] = int(np.sum(lam - cf1 < -thr))
    d["nNoise"] = int(np.sum((lam - cf1 < 0.0) & (lam - cf1 >= -thr)))
    d["thr"] = thr
    scmax = float(np.max(np.abs(sc["coef"])))
    rts = np.array([], dtype=float)
    d["resid"] = np.nan
    d["nroots"] = 0
    if scmax > 1e-14:
        r = np.roots(sc["coef"])
        ok = (np.abs(r.imag) < 1e-9) & (r.real > 0.0)
        if ok.any():
            rts = np.sort(r[ok].real)
            d["nroots"] = int(len(rts))
            d["resid"] = float(np.max(np.abs(np.polyval(sc["coef"], rts))) / scmax)
    else:
        d["resid"] = 0.0
    d["xcubic"] = float(rts[0]) if len(rts) else np.nan
    f = lam - cf1
    d["mindev"] = float(np.min(f))
    d["theta"] = np.nan
    if (f < -thr).any():                       # descent must be REAL (above the threshold) to enter,
        negidx = np.nonzero(f < -thr)[0]        # but the bisection targets the exact zero
        after = [i for i in negidx if (i + 1 < len(TH) and f[i + 1] > 0.0)]
        if after:
            lo, hi = TH[after[0]], TH[after[0] + 1]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if lam1_at(A0, B2, v1, u, mid) - cf1 < 0.0:
                    lo = mid
                else:
                    hi = mid
            d["theta"] = 0.5 * (lo + hi)
    d["xhi"] = (2.0 * sc["g"] / sc["P"]) if sc["P"] > 0 else np.nan
    d["gst"] = (sc["g"] / sc["P"]) if sc["P"] > 0 else np.nan
    # SIGNED drop: b49's dip is -L1^2/(4 L2) < 0, so the prediction must carry the same sign
    d["pred_dip"] = (-abs(sc["drdt"]) * sc["g"] ** 2 / (c2 * sc["P"])) if sc["P"] > 0 else np.nan
    if np.isfinite(d["xhi"]) and d["xhi"] > 0:
        d["corr3"] = (d["xhi"] ** 2) * sc["kap"] * sc["D"] ** 2 / sc["P"]     # predicted delta of x/x_hi-1
    # NOISE FLOOR of the bisection, measured not assumed: the crossing is shallow, so a solver error of
    # ~1e-16 in lambda shows up as a LARGE error in theta. implied = |slope| * |theta* - atan(x_cubic)|
    # is what the two routes disagree by IN THE SOLVER'S OWN UNITS. Registered: implied < 1e-14.
    d["slop"] = np.nan
    d["implied"] = np.nan
    if np.isfinite(d["theta"]) and d["theta"] > 0 and np.isfinite(d["xcubic"]):
        h = 0.05 * d["theta"]
        fa = lam1_at(A0, B2, v1, u, d["theta"] + h) - cf1
        fb = lam1_at(A0, B2, v1, u, d["theta"] - h) - cf1
        d["slop"] = abs(fa - fb) / (2.0 * h)
        d["implied"] = d["slop"] * abs(d["theta"] - float(np.arctan(d["xcubic"])))
    return d


def stats(v, scale=1.0):
    v = np.asarray(v, dtype=float) * scale
    v = v[np.isfinite(v)]
    if not len(v):
        return "n/a"
    return f"med={np.median(v):.4e} min={v.min():.3e} max={v.max():.3e} n={len(v)}"


def run_cell(tag, gen, n=N, do_rand=True):
    rows, nstat, nskip = [], 0, 0
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            nskip += 1
            continue
        rr = ray_L(s)
        if rr is None:
            nstat += 1
            d = None
            for j in range(3):                     # equal-angle falsifier on THREE independent rays,
                w = rng.normal(size=s["A0"].shape[0])   # not on one lucky choice of u
                w = w - s["v1"] * (s["v1"] @ w)
                nw = float(np.linalg.norm(w))
                if nw < 1e-10:
                    continue
                e = ray_check(s, w / nw)
                if d is None:
                    d = dict(e)
                    d["belowRand"] = 0
                    d["nDescRand"] = 0
                    d["nNoiseRand"] = 0
                d["eRand"] = max(d.get("eRand", 0.0), e["eB"], e["eC"], e["eT"], e["crossrel"],
                                 e["resid"] if np.isfinite(e["resid"]) else 0.0)
                d["belowRand"] += e["below"]
                d["nDescRand"] += e["nDesc"]
                d["nNoiseRand"] += e["nNoise"]
            if d is None:
                continue
            d.update(stat=1, m0=s["m0"], cf1=s["cf1"], gap=s["cf1"] - s["true"],
                     ang=float(np.arccos(np.clip(abs(s["vstar"] @ s["v1"]), -1, 1))),
                     l1vstar=float(crossings(np.array([float(s["vstar"] @ s["A0"] @ s["vstar"])]),
                                             np.array([float(s["vstar"] @ s["B2"] @ s["vstar"])]))[0]))
            rows.append(d)
            continue
        u, nrm, L1, L2, tp, dip = rr
        d = ray_check(s, u)
        d.update(stat=0, m0=s["m0"], cf1=s["cf1"], gap=s["cf1"] - s["true"], L1=L1, L2=L2, tp=tp,
                 dip=dip, ratioL1=L1 / (d["drdt"] * d["A1"]), ratioL2=L2 / (d["drdt"] * d["A2"])
                 if d["A2"] != 0 else np.nan)
        d["r_bisect"] = d["theta"] / np.arctan(d["xcubic"]) - 1.0 if np.isfinite(d["theta"]) \
            and np.isfinite(d["xcubic"]) else np.nan
        d["r_xhi"] = d["theta"] / np.arctan(d["xhi"]) - 1.0 if np.isfinite(d["theta"]) \
            and np.isfinite(d["xhi"]) and d["xhi"] > 0 else np.nan
        d["r_tp"] = tp / d["gst"] - 1.0 if np.isfinite(tp) and np.isfinite(d["gst"]) else np.nan
        d["r_corr"] = d["r_xhi"] / d["corr3"] - 1.0 if np.isfinite(d["r_xhi"]) \
            and np.isfinite(d.get("corr3", np.nan)) and abs(d["corr3"]) > 1e-18 else np.nan
        d["r_dip"] = dip / d["pred_dip"] - 1.0 if np.isfinite(dip) and np.isfinite(d["pred_dip"]) \
            and abs(d["pred_dip"]) > 0 else np.nan
        d["ang"] = float(np.arccos(np.clip(abs(s["vstar"] @ s["v1"]), -1, 1)))
        d["dipfrac"] = dip / d["gap"] if np.isfinite(dip) else np.nan
        d["predipfrac"] = d["pred_dip"] / d["gap"] if np.isfinite(d["pred_dip"]) else np.nan
        vv = s["vstar"]
        d["l1vstar"] = float(crossings(np.array([float(vv @ s["A0"] @ vv)]),
                                       np.array([float(vv @ s["B2"] @ vv)]))[0])
        if do_rand:
            d["belowRand"] = 0
            d["nDescRand"] = 0
            d["nNoiseRand"] = 0
            for j in range(3):
                w = rng.normal(size=s["A0"].shape[0])
                w = w - s["v1"] * (s["v1"] @ w)
                nw = float(np.linalg.norm(w))
                if nw < 1e-10:
                    continue
                e = ray_check(s, w / nw)
                d["eRand"] = max(d.get("eRand", 0.0), e["eB"], e["eC"], e["eT"], e["crossrel"],
                                 e["resid"] if np.isfinite(e["resid"]) else 0.0)
                d["belowRand"] = d.get("belowRand", 0) + e["below"]
                d["nDescRand"] += e["nDesc"]
                d["nNoiseRand"] += e["nNoise"]
        rows.append(d)
    if not rows:
        print(f"  {tag:>26}: no usable draw (skipped {nskip})")
        return None
    a = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    sm = lambda k: a(k)[stat == 1]           # numbers from the EQUAL-ANGLE / stationary draws only
    st = lambda k: a(k)[stat == 0]           # numbers from the generic (non-stationary) draws only
    stat = np.array([r["stat"] for r in rows])
    print(f"  {tag:>26}: n={len(rows):3d} stationary={int(stat.sum())} skipped={nskip}"
          f"  dimE0={sorted({r['m0'] for r in rows})}")
    print(f"  {'':>26}  [T0] |v1'A0u|/c2 {stats(a('crossrel'))}  eB {stats(a('eB'))}"
          f"  eC {stats(a('eC'))}  eT {stats(a('eT'))}  cubic-resid {stats(a('resid'))}"
          f"  rand-u worst {stats(a('eRand'))}")
    print(f"  {'':>26}  [T1] theta*/atan(x_cubic)-1 {stats(a('r_bisect'))}"
          f"  theta* {stats(a('theta'))}  x_hi=2g/P {stats(a('xhi'))}"
          f"  theta*/atan(x_hi)-1 {stats(a('r_xhi'))}"
          f"  #(cubic has no positive root)={int(np.sum(a('nroots') == 0))}"
          f"  #(no bisected crossing)={int(np.sum(~np.isfinite(a('theta'))))}"
          f"  #(root exists but no crossing)={int(np.sum((a('nroots') > 0) & ~np.isfinite(a('theta'))))}")
    print(f"  {'':>26}  [T1b] THIRD-ORDER relative correction of x_hi: predicted kappa*D^2*x_hi^2/P "
          f"{stats(a('corr3'))}  measured theta*/atan(x_hi)-1 {stats(a('r_xhi'))}"
          f"  ratio measured/pred-1 {stats(a('r_corr'))}")
    print(f"  {'':>26}  [T1c] BISECTION NOISE FLOOR (registered: implied < 1e-14):"
          f" |dlambda/dtheta| at theta* {stats(a('slop'))}"
          f"  implied lambda-mismatch between the two exact routes {stats(a('implied'))}"
          f"  => theta-level disagreement is solver noise, not an iff exception, iff implied << gap")
    # SELF-CONSISTENCY of the [T1c] column itself: implied must equal slop*|theta-atan(xcubic)| by
    # construction, so it is bounded by slop_max * r_bisect_max * atan(xcubic)_max. If the printed max
    # breaks that bound, the DEFECT IS IN MY CHECK, not in the cubic (rule 9).
    trip = [(r.get("implied", np.nan), r.get("slop", np.nan), r.get("theta", np.nan),
             r.get("xcubic", np.nan), r.get("r_bisect", np.nan)) for r in rows]
    bad = [t for t in trip if all(np.isfinite(x) for x in t[:4])]
    eid = max((abs(t[0] - t[1] * abs(t[2] - np.arctan(t[3]))) for t in bad), default=np.nan)
    wr = max(bad, key=lambda t: t[0], default=None)
    bnd = (np.nanmax([t[1] for t in bad]) * np.nanmax([abs(t[4]) for t in bad])
           * np.nanmax([abs(np.arctan(t[3])) for t in bad])) if bad else np.nan
    print(f"  {'':>26}      [T1c-selfcheck] max |implied - slop*|theta-atan(xc)||={eid:.2e}"
          f"  worst implied={wr[0]:.3e} (slop={wr[1]:.3e} theta={wr[2]:.4e} xcubic={wr[3]:.6e}"
          f" r_bisect={wr[4]:.3e})  bound slop_max*r_max*xc_max={bnd:.3e}"
          f"  {'CONSISTENT' if np.isfinite(bnd) and wr is not None and wr[0] <= 10*bnd else 'INCONSISTENT'}")
    print(f"  {'':>26}  [T2] L1/(drdt*A1)-1 {stats(a('ratioL1') - 1.0)}"
          f"  L2/(drdt*A2)-1 {stats(a('ratioL2') - 1.0)}"
          f"  dip b49/pred-1 {stats(a('r_dip'))}  tp/(g/P)-1 {stats(a('r_tp'))}")
    print(f"  {'':>26}  [T3] EQUAL-ANGLE FALSIFIER (stationary draws ONLY: {int(stat.sum())} draws"
          f" x 3 random u)  #(lambda_1 < cf1 - floor)={int(np.nansum(sm('belowRand')))}"
          f"  #(grid points with a real descent)={int(np.nansum(sm('nDescRand')))}"
          f"  deepest apparent dip {stats(sm('mindev'))}"
          f"  gap {stats(sm('gap'))}  thr=1e-12*gap {stats(sm('thr'))}"
          f"  #(g<=1e-12)={int(np.sum(sm('g') <= 1e-12))}/{max(1, int(stat.sum()))}")
    print(f"  {'':>26}      generic draws, STEEPEST ray (context, NOT the falsifier):"
          f" #(below-floor)={int(np.nansum(st('below')))}"
          f"  #(real descent)={int(np.nansum(st('nDesc')))}"
          f"  #(round-off band)={int(np.nansum(st('nNoise')))}"
          f"  deepest dip {stats(st('mindev'))}"
          f"  #(g<=1e-12)={int(np.sum(st('g') <= 1e-12))}/{max(1, int((stat == 0).sum()))}")
    print(f"  {'':>26}  [T6] #(P<=0)={int(np.sum(a('P') <= 0.0))} #(L2<=0)={int(np.sum(a('L2') <= 0.0))}"
          f"  #(no upper crossing)={int(np.sum(~np.isfinite(a('theta')) & (stat == 0)))}"
          f"  #(P>0 but no crossing)={int(np.sum((a('P') > 0) & ~np.isfinite(a('theta')) & (stat == 0)))}")
    print(f"  {'':>26}  [T5] angle(v*,v1) {stats(a('ang'))}"
          f"  angle/theta* {stats(a('ang') / np.where(a('theta') > 0, a('theta'), np.nan))}"
          f"  gapfrac(lambda_1(v*)) {stats((a('cf1') - a('l1vstar')) / a('gap'))}")
    return rows


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


print("=" * 130)
print("[T0-T6] the descent-set boundary is a CUBIC in tan(theta); its leading root 2g/P is b52's t_hi.")
print("Main cells = b53/b55's draws (seed 60311, same order); the ladder switches to seed 51400.")
for name, cs in SPECTRA.items():
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix (DETERMINISTIC: 40 replicates)"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix (DETERMINISTIC: 40 replicates)")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)),
                                                              np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(x))
                                                              for x in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 130)
print("[T4] INTERPOLATION LADDER -- own seed 51400 (switched AFTER the shared-draw section above).")
print("     cosines (20-d, 20, 20+d) in DEGREES.  Registered: x_hi, g, P, dip/gap all slope 0;")
print("     D slope 1.  (My first instinct was x_hi ~ d; the derivation of A_0 = Q2 diag(cos^2) Q2^T")
print("     says v_1 does NOT move with d, so the layer should NOT collapse.  Both branches informative.)")
rng = np.random.default_rng(51400)
ladder = {}
for deg in (1e-3, 1e-2, 1e-1, 1.0, 5.0):
    cs = [np.cos(np.deg2rad(20.0 - deg)), np.cos(np.deg2rad(20.0)), np.cos(np.deg2rad(20.0 + deg))]
    ladder[deg] = run_cell(f"q=3 equal-angled d={deg:g}deg",
                           lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)),
                           n=30, do_rand=True)

print("\n" + "=" * 130)
print("[T4b] log-log slopes of the ladder medians (the registered test is on THESE numbers):")
med = lambda rows, k: float(np.median([r[k] for r in rows
                                       if r is not None and np.isfinite(r.get(k, np.nan))]))
xs = np.array([d for d in sorted(ladder) if ladder[d]], dtype=float)
BANDS = {"xhi": (-0.2, 0.2), "g": (-0.2, 0.2), "P": (-0.2, 0.2), "D": (0.7, 1.3),
         "dipfrac": (-0.2, 0.2)}
if len(xs) >= 3:
    for key in ("xhi", "g", "P", "D", "dipfrac"):
        ys = np.array([med(ladder[d], key) for d in xs], dtype=float)
        ok = np.isfinite(ys) & (ys > 0)
        if ok.sum() >= 3:
            sl = float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])
            lo, hi = BANDS[key]
            print(f"   slope log({key}) vs log(d) = {sl:+.4f}  band [{lo},{hi}]"
                  f"  {'PASS' if lo <= sl <= hi else 'FALSIFIED'}"
                  f"  vals={np.array2string(ys[ok], precision=3)}")
        else:
            print(f"   slope log({key}): only {int(ok.sum())} positive finite medians -- not fit, reported")
else:
    print("   ladder produced fewer than 3 usable cells")

print("\n" + "=" * 130)
print("判读（预登记，跑前写）")
print("  T0 恒等式门（射线闭式 vs 直接计算、T 两种算法、立方在自身根上的残差、v1'A0u=0）：任一项 >1e-10")
print("     就把后面全部作废 —— 这是 (p) 号撤回的失败模式，所以先跑并且逐抽样打印。")
print("     几何网格（1e-10..pi/2，3000 点）是按规则 11 选的：b51 的 linspace 左端点就是统计量的天花板，")
print("     而边界层宽度可到 1e-5，均匀网会整段漏掉下降区。")
print("  T1 二分出的精确边界 vs 立方根：两者都是**精确量**，比值应在 1e-8 内。>1e-6 则 lem:margin 的 iff")
print("     在某个方向上有例外（最贵失败，参照物是独立求根器，不是我自己的代数）。")
print("     x_hi=2g/P 与 tp=g/P 是**截断**，只报比值不登记方向；它们与精确值的偏离量本身是信息。")
print("  T2 跨道核对 L1：手算已证 F_lambda = c^2(2t1mu-1-t1)、F_C = -mu(1-mu)、c1 = 2g，于是 b49 的 L1")
print("     与 (dr/dt)A1 恒等，比值 >1e-6 说明章节里 rem:chord 的 dip 公式与 lem:margin 有一个是错的。")
print("     L2 与 dip 的比值**不预登记符号**：差的正是 r 的二阶项。")
print("  T3 等角端点：D=g=0，立方成 P x(x^2+1)=0 且 P=t1c^2-b>0 ⇒ 唯一实根 x=0，下降边界就坐在 v1 上，")
print("     也就是下降集为空。证伪器 = 任一 theta 使 lambda_1 < cf_1 - floor。")
print("     另：一般格里 #(g<=1e-12) 必须为 0，否则 x_hi=0 是平凡成立而非结构。")
print("  T4 插值（§48.11(b)）：我**先猜** x_hi ~ d，推导之后改成 x_hi 斜率 0（v1 由 A_0 的最大余弦选定，")
print("     与 d 无关；d=0 那一点 dimE0 跳到 q、选取规则换成 C-极大者，于是 g=0 —— 等角是孤立点，")
print("     不是极限）。注册的是斜率带；若 x_hi 斜率 >0.5，说明层真的塌缩，§48.11(b) 以连续陈述重开。")
print("     两条分支都是信息（规则 6），我不挑好看的那条。")
print("  T5 只测不猜：angle(v*,v1)/theta*。b53 说两个量级相同；比值不是 O(1) 则 §47.3 的「同一个量的")
print("     第二次计算」只是量级巧合，机制没接上，照直说。")
print("  T6 只数不平：#(P<=0)、#(L2<=0)、两者是否同侧、以及 #(扫描找不到上边界)。B-CLAIM-56(i) 要的")
print("     L2>0 在此归约为标量不等式 kappa(c^2-a)c^2 > b - t1 a；q=2 时垂直方向唯一 ⇒ 逐抽样即穷尽。")
