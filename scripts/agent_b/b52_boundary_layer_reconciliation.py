"""
b52 -- why does b51's 2080-8256-POINT direction net find NO improvement of lambda_1 over cf1 (median
gapfrac 0.0000 in seven of eight cells) while b50, on the same cell generators and the same denominator
cf1 - lam_min, reports that b49's analytic ray removes 0.13-0.27 of that gap at its median?  Both
numbers are computed with the SAME function `crossings`, so at most one of them can be about the same
object.  One of the two readings has to be a mistake of mine, and the difference between them is a
property of the DESCENT SET of v -> lambda_1(v) at v1:

  HYPOTHESIS (the boundary layer).  b49's t* = -L1/(2 L2) is a ratio of two jet coefficients, and the
  dip -L1^2/(4 L2) is a LARGE FRACTION OF A SMALL gap: the gap cf1 - lam_min is often 1e-6..1e-3, so the
  value of lambda_1 can fall below cf1 only for t inside a thin interval (0, t_hi) around the ray, and
  return above cf1 by the time t reaches b51's first off-axis grid point, pi/(2*64) = 2.46e-2.  A
  uniform t-grid from 1e-3 to pi/2 then MISSES the whole descent set, and the net's minimum is exactly
  the t = 0 entry -- which is what b51 printed.  If this is right, "min over a direction net" is not a
  measurement of the direction knob at all: it is a statement about my grid (memory rule 7 -- a net is
  not an optimiser -- and rule 5, sampled vs exhaustive).

  ALTERNATIVE (the grid is fine and b51 is broken).  If, on the exact ray direction u, b51's own t-list
  DOES find lambda_1 < cf1, then the boundary layer is a fiction and the missing piece is in my u-net
  (perp_net / broadcasting), which would void b51's [C] and [D] and force a retraction.

THE CONSTRUCTION adds nothing new: v1 and the ray u come from b49/b50's two-jet formula (copied), the
moments m_k(v) and the n=2 cubic are b50/b51's, and every direction is unit by orthogonality (v = cos t
v1 + sin t u with u _|_ v1, ||u|| = ||v1|| = 1, so ||v|| = 1 WITHOUT a normalisation step -- which is
itself a check, printed below).

PRE-REGISTERED, falsifiers as numbers:
  A  SIDE, for every new object: min over the fine ray profile of lambda_1 and of lambda_2, and the
     rotated-tangent points, must all be >= lam_min(G) up to the per-draw floor.  FALSIFIER: any
     violation. Then the root filter is picking a pole-crossing and nothing below is evidence.
  B  RECONCILIATION.  On the SAME draws as b51 (same seed, same call order):
        (i) min over a 300-point GEOMETRIC profile t in [1e-8, pi/2] along u_ray, of lambda_1, must
            reproduce b50's cf2 to ~1e-12 (same object, two routes);
        (ii) min over b51's t-list (0, then 64 points from 1e-3 to pi/2) on that SAME u must equal cf1
            for MOST draws.  FALSIFIER of the boundary-layer reading: if the b51 t-list on the exact ray
            finds the descent, the bug is in my u-net, not in the grid, and b51 must be retracted.
  C  WIDTH.  t_hi(lambda_1) := the largest grid t with lambda_1(v(t)) < cf1.  Prediction: median t_hi
     < 2.46e-2 (b51's coarse spacing) so that a uniform net of ~2000 directions cannot see it.
     FALSIFIER: median t_hi > 1e-1.
  D  THE ROUTE QUESTION, and the reason this file exists.  Compare the two knobs by GRID SENSITIVITY:
     gapfrac of min over the polar net as a function of the number of t-points nt = 4, 8, 16, 32, 64,
     128, for lambda_1 and for lambda_2.  Prediction: lambda_1's net gain stays ~0 at every nt (its
     descent is a boundary layer in t, invisible to any uniform grid), while lambda_2's net gain is
     ALREADY POSITIVE at nt = 4 or 8 and grows smoothly -- i.e. the n=2 improvement over the directions
     is an OPEN-SET effect, not a razor.  FALSIFIER (against publishing a direction search for n=2): if
     lambda_2's net gain also requires nt >= 32 to appear, its b51 gain is the same grid artifact and the
     honest recommendation is "report lambda_2(v1) only, with no direction claim at any nt".
  E  WIDTH IN u.  Fix t = t*, rotate u by delta in v1^perp (only possible for q >= 3).  Report the
     fraction of the descent retained at delta = 1e-2, 0.05, 0.2. Prediction: lambda_1's descent is a
     CONE of half-angle at most ~0.05 rad (b51 used 32 u's, spacing 0.20 rad), so the net misses it in
     u as well as in t.  FALSIFIER: retention > 0.9 at delta = 0.2, meaning u-resolution was never the
     problem and only t was.
"""
import numpy as np

rng = np.random.default_rng(60311)          # b51's seed: the draws must be the SAME objects
NET = np.random.default_rng(9007)
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
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = float(np.linalg.eigvalsh(Gr)[0])
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
    mu, W = np.linalg.eigh(E0.T @ B2 @ E0)
    t1 = float(mu[-1]) / c2
    cf1 = cf_of(t1, c2)
    v1 = E0 @ W[:, -1]
    v1 = v1 / np.linalg.norm(v1)
    if cf1 >= min(float(ev[0]), 1.0):
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, cf1=cf1, v1=v1, m0=E0.shape[1], M=Mm,
                floor=100.0 * EPS * np.linalg.norm(Gr, 2))


def ray(s):
    """b49/b50's two-jet ray, verbatim, but returning (u, t*) as well so the profile can be measured."""
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
        return None
    return u, -L1 / (2.0 * L2), L1, L2


def lam2_batch(VV, M):
    m = np.stack([np.einsum("ij,jk,ik->i", VV, Mk, VV) for Mk in M])
    m0, m1, m2, m3 = m
    a0 = m1 / m0
    b1 = m2 / m0 - a0 * a0
    degen = ~(b1 > 0.0)
    b1s = np.where(degen, 1.0, b1)
    a1 = (m3 - 2.0 * a0 * m2 + a0 * a0 * m1) / (m0 * b1s)
    s1 = a0 + a1
    p2 = np.where(degen, 1.0, a0 * a1 - b1s)
    c3, c2c, c1c, c0 = p2, -(a0 * a1 - b1s + s1), (s1 + 1.0) - (m0 * s1 - m1), m0 - 1.0
    C = np.zeros((len(VV), 3, 3))
    C[:, 0, 2] = -c0 / c3
    C[:, 0, 1] = -c1c / c3
    C[:, 0, 0] = -c2c / c3
    C[:, 1, 0] = 1.0
    C[:, 2, 1] = 1.0
    rts = np.linalg.eigvals(C)
    ok = (np.abs(rts.imag) < 1e-8) & (rts.real >= 0.0)
    rts = np.where(ok, rts.real, np.nan)
    out = np.min(np.where(np.isnan(rts), np.inf, rts), axis=1)
    den = 1.0 - s1 * out + p2 * out * out
    res = (m0 - (m0 * s1 - m1) * out) - (1.0 - out) * den
    bad = ~(den > 0.0) | (np.abs(res) > 1e-6 * np.maximum(1.0, m0)) | ~np.isfinite(out)
    out = np.where(degen | bad, np.nan, out)
    if degen.any():
        out = np.where(degen, crossings(m0, m1), out)
    return out


def polar(v1, u, ts):
    """cos t v1 + sin t u for a LIST of t; u must be unit and orthogonal to v1, in which case the norms
    are 1 without any normalisation -- the printed max deviation is that check."""
    V = np.cos(ts)[:, None] * v1[None, :] + np.sin(ts)[:, None] * u[None, :]
    return V, float(np.max(np.abs(np.linalg.norm(V, axis=1) - 1.0)))


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


NTS = (4, 8, 16, 32, 64, 128)
TS51 = np.concatenate(([0.0], np.linspace(1e-3, 0.5 * np.pi, 64)))   # b51's t-list, verbatim
TSPROF = np.geomspace(1e-8, 0.5 * np.pi, 300)
SPACING51 = 0.5 * np.pi / 64


def run_cell(tag, gen, n=N):
    acc = {k: [] for k in ("g_ray", "g_prof", "g51", "thi1", "thi2", "gray2", "g512")}
    dims, viol = [], dict(prof1=0, prof2=0, rot=0, net1=0, net2=0, netg=0)
    nuniv, nq2, nq2rot, norms, sat = 0, 0, 0, 0.0, []
    ret = {d: [] for d in (1e-2, 0.05, 0.2)}
    bynt = {nt: {"l1": [], "l2": [], "l1g": [], "l2g": []} for nt in NTS}
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        v1, M, cf1, true, floor = s["v1"], s["M"], s["cf1"], s["true"], s["floor"]
        gap = cf1 - true
        if not (gap > floor):
            continue
        r = ray(s)
        if r is None:
            nuniv += 1
            continue
        u, tp = r[0], r[1]
        # ---- B(i)/C: the fine geometric profile along the exact ray
        V, nn = polar(v1, u, TSPROF)
        norms = max(norms, nn)
        m1 = np.einsum("ij,jk,ik->i", V, M[0], V)
        m1b = np.einsum("ij,jk,ik->i", V, M[1], V)
        l1p = crossings(m1, m1b)
        l2p = lam2_batch(V, M)
        # B(i): b50's cf2 is lambda_1 evaluated at EXACTLY t*, so it must agree with the profile min to
        # the profile's own resolution (300 geometric points), not to 1e-12 -- recorded here so the
        # reconciliation claim is stated at the strength the data supports.
        base = np.cos(tp) * v1 + np.sin(tp) * u
        l1ray = float(crossings(np.array([base @ M[0] @ base]), np.array([base @ M[1] @ base]))[0])
        acc["g_ray"].append((cf1 - l1ray) / gap)
        dims.append(s["m0"])
        acc["g_prof"].append((cf1 - np.nanmin(l1p)) / gap)
        acc["gray2"].append((cf1 - np.nanmin(np.where(np.isfinite(l2p), l2p, np.inf))) / gap)
        # t_hi = the largest t on the profile with a strict decrease (0 if none); for lambda_2 the base
        # value is lambda_2(v1), not cf1, because that is the quantity b51's test C compares against.
        l2_0 = float(lam2_batch(v1[None, :], M)[0])
        below1 = np.where(l1p < cf1 - floor, TSPROF, 0.0)
        below2 = np.where(np.isfinite(l2p) & (l2p < l2_0 - floor), TSPROF, 0.0)
        acc["thi1"].append(float(below1.max()))
        acc["thi2"].append(float(below2.max()))
        viol["prof1"] += int(np.nanmin(l1p) < true - floor)
        viol["prof2"] += int(np.nanmin(np.where(np.isfinite(l2p), l2p, np.inf)) < true - floor)
        # ---- B(ii): b51's t-list on the SAME u -- does the coarse grid see the descent?
        V51, nn51 = polar(v1, u, TS51)
        l51 = crossings(*[np.einsum("ij,jk,ik->i", V51, Mk, V51) for Mk in M[:2]])
        l512 = lam2_batch(V51, M)
        acc["g51"].append((cf1 - np.min(l51)) / gap)
        acc["g512"].append((cf1 - np.nanmin(np.where(np.isfinite(l512), l512, np.inf))) / gap)
        # ---- D: grid sensitivity of BOTH knobs on the full polar net (u from a 32-point circle, or
        # Gaussian rows for q>=4, exactly as b51 assembles it) -- measured as nt varies.
        q = len(v1)
        if q == 2:
            nq2 += 1
            Us = np.stack([u, -u])
        elif q == 3:
            w = np.cross(v1, u)
            w = w / np.linalg.norm(w)
            ph = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            Us = np.stack([np.cos(p) * u + np.sin(p) * w for p in ph])
        else:
            Us = _rows([v1], q, 32)
        # the u-SET is fixed OUTSIDE the nt loop: test D compares grids in t, so if u also changed
        # between nt values a difference could not be attributed to t-resolution at all.
        for nt in NTS:
            ts = np.concatenate(([0.0], np.linspace(1e-3, 0.5 * np.pi, nt)))
            W = (np.cos(ts)[:, None, None] * v1[None, None, :] +
                 np.sin(ts)[:, None, None] * Us[None, :, :]).reshape(-1, q)
            W = W / np.linalg.norm(W, axis=1, keepdims=True)
            a1 = crossings(*[np.einsum("ij,jk,ik->i", W, Mk, W) for Mk in M[:2]])
            a2 = lam2_batch(W, M)
            bynt[nt]["l1"].append((cf1 - np.min(a1)) / gap)
            bynt[nt]["l2"].append((cf1 - np.nanmin(np.where(np.isfinite(a2), a2, np.inf))) / gap)
            # D2, ADDED AFTER THE FIRST RUN because D came out FLAT in nt: linspace(1e-3, pi/2, nt)
            # contains its left endpoint 1e-3 for EVERY nt, so the minimum over the net was already
            # attained at the smallest off-axis angle in the list and nt cannot change it -- my
            # sensitivity test was degenerate by construction, which is itself a finding about b51.
            # The honest variant is a grid that reaches INSIDE the layer, at the same cost in points.
            tsg = np.concatenate(([0.0], np.geomspace(1e-6, 0.5 * np.pi, nt)))
            Wg = (np.cos(tsg)[:, None, None] * v1[None, None, :] +
                  np.sin(tsg)[:, None, None] * Us[None, :, :]).reshape(-1, q)
            Wg = Wg / np.linalg.norm(Wg, axis=1, keepdims=True)
            a1g = crossings(*[np.einsum("ij,jk,ik->i", Wg, Mk, Wg) for Mk in M[:2]])
            a2g = lam2_batch(Wg, M)
            bynt[nt]["l1g"].append((cf1 - np.min(a1g)) / gap)
            bynt[nt]["l2g"].append((cf1 - np.nanmin(np.where(np.isfinite(a2g), a2g, np.inf))) / gap)
            viol["net1"] += int(np.min(a1) < true - floor)
            viol["net2"] += int(np.nanmin(np.where(np.isfinite(a2), a2, np.inf)) < true - floor)
            viol["netg"] += int(min(np.min(a1g), np.nanmin(np.where(np.isfinite(a2g), a2g, np.inf)))
                                < true - floor)
            if nt == max(NTS):
                sat.append((np.nanmin(np.where(np.isfinite(a2g), a2g, np.inf)) - true) / gap)
        # ---- E: rotation width in u at fixed t = tp
        if q >= 3:
            w = np.cross(v1, u) if q == 3 else _rows([v1, u], q, 1)[0]
            w = w / np.linalg.norm(w)
            base = np.cos(tp) * v1 + np.sin(tp) * u
            lb = float(crossings(np.array([base @ M[0] @ base]), np.array([base @ M[1] @ base]))[0])
            for d in (1e-2, 0.05, 0.2):
                vr = np.cos(tp) * v1 + np.sin(tp) * (np.cos(d) * u + np.sin(d) * w)
                lr = float(crossings(np.array([vr @ M[0] @ vr]), np.array([vr @ M[1] @ vr]))[0])
                ret[d].append(np.clip((cf1 - lr) / max(cf1 - lb, 1e-300), -9.0, 9.0))
                viol["rot"] += int(lr < true - floor)
        else:
            nq2rot += 1
    if not acc["g_ray"]:
        print(f"   {tag:>26}: no usable draw")
        return
    g = lambda k: np.array([x for x in acc[k] if np.isfinite(x)])
    print(f"   {tag:>26}: n={len(acc['g_ray']):3d} dimE0={sorted(set(dims))}"
          f" no-ray(stationary or L2<=0)={nuniv} q=2(no u-rotation)={nq2rot}"
          f"   [A] viol prof:{viol['prof1'] + viol['prof2']} net:{viol['net1'] + viol['net2']}"
          f" net-geom:{viol['netg']} rot:{viol['rot']}"
          f"   max||v||-1| on the un-normalised polar form = {norms:.1e}")
    print(f"{'':>28}  [B] gapfrac lambda_1: at t* exactly (b50's cf2) med={np.median(g('g_ray')):.4f}"
          f"   300-pt fine profile med={np.median(g('g_prof')):.4f}"
          f"   b51-t-list-on-the-same-ray med={np.median(g('g51')):.4f}"
          f"   (=0 => the coarse grid is blind, b51's net story confirmed)"
          f"   [C] t_hi(lambda_1): med={np.median(g('thi1')):.2e} p90={np.percentile(g('thi1'), 90):.2e}"
          f" vs spacing51={SPACING51:.2e}   draws with t_hi < spacing51 = "
          f"{int(np.sum(g('thi1') < SPACING51))}/{len(g('thi1'))}")
    print(f"{'':>28}  [C2] t_hi(lambda_2): med={np.median(g('thi2')):.2e}"
          f" p90={np.percentile(g('thi2'), 90):.2e}   draws with t_hi < spacing51 ="
          f" {int(np.sum(g('thi2') < SPACING51))}/{len(g('thi2'))}"
          f"   gapfrac lambda_2 on the ray med={np.median(g('gray2')):.4f}"
          f"   b51-t-list on the ray, lambda_2 med={np.median(g('g512')):.4f}")
    print(f"{'':>28}  [D] gapfrac(min over the 32u x nt net): columns are the UNIFORM t-list"
          f" (1e-3..pi/2) | the GEOMETRIC list (1e-6..pi/2), same number of points:")
    for nt in NTS:
        a = np.array(bynt[nt]["l1"])
        b = np.array(bynt[nt]["l2"])
        ag = np.array(bynt[nt]["l1g"])
        bg = np.array(bynt[nt]["l2g"])
        print(f"{'':>34}  nt={nt:4d}  l1 {np.median(a):7.4f} | {np.median(ag):7.4f}"
              f"     l2 {np.median(b):7.4f} | {np.median(bg):7.4f}")
    sa = np.array(sat)
    print(f"{'':>28}  [D2] saturation: (min_net lambda_2 - lam_min)/gap on the geometric grid,"
          f" nt=128: med={np.median(sa):.2e} max={np.max(sa):.2e}"
          f"   (this is HOW FAR the best found member still sits above the certificate;"
          f" 1e-3 means the gap is removed to a per-mille)")
    if ret[1e-2]:
        print(f"{'':>28}  [E] fraction of the ray descent retained after rotating u by delta:"
              + "".join(f"   d={d:g} med={np.median(np.array(ret[d])):6.3f}"
                        f" p10={np.percentile(np.array(ret[d]), 10):6.3f}"
                        for d in (1e-2, 0.05, 0.2))
              + f"   (b51's u-spacing was 2pi/32 = 0.20 rad)")
    else:
        print(f"{'':>28}  [E] no q>=3 draw to rotate in")


def _gram(x, *refs):
    for r in refs:
        x = x - r * (x @ r) / (r @ r)
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else None


def _rows(refs, q, nu):
    """nu unit rows orthogonal to every vector in refs (a None from _gram is just retried)."""
    out = []
    while len(out) < nu:
        g = _gram(NET.normal(size=q), *refs)
        if g is not None:
            out.append(g)
    return np.stack(out)


print("=" * 124)
print("[A-E] reconciliation of b51 (uniform direction net finds NOTHING for lambda_1) with b50")
print("      (the analytic ray removes 0.13-0.27 of cf1's gap).  Hypothesis: the descent set of")
print("      lambda_1 at v1 is a boundary layer of width t_hi << pi/(2*64) = 2.46e-2 in t, so no")
print("      uniform grid of ~2000 directions resolves it -- whereas lambda_2's gain is an open-set")
print("      effect.  Test D is the route question: if lambda_2's net gain also needs nt>=32, the")
print("      b51 gain is a grid artifact and there is no direction search to advertise at any n.")
for name, cs in SPECTRA.items():
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 124)
print("判读（预登记，跑前写）：")
print("  A 侧面：细剖面、旋转点、各 nt 网格上的 min 都不得低于 lam_min(G)（超 floor 即违反）。")
print("     任何违反 => 三次方程根筛选挑了假根，本文件的 D 结论作废，回到 b51 的 C 也未决。")
print("  B 对账：同一批抽样（同 seed、同调用顺序）上，沿解析射线 u 的 300 点几何级数剖面必须复现 b50")
print("     的 cf2；而 b51 的 t 列（0, 然后 64 点 1e-3..pi/2）在同一 u 上的 min 若'恰好等于 cf1'，")
print("     就说明 b51 不是算错，而是网格看不见下降区。反向：若该 t 列能找到下降，b51 的 u 网有 bug，")
print("     必须撤回其 [C]/[D]。")
print("  C 宽度：t_hi(lambda_1) 中位数 < 2.46e-2（b51 的 t 间距）才叫边界层；> 1e-1 则假设死。")
print("  D 路线：把两个旋钮按'网格敏感度'对比 —— lambda_1 的网增益应随 nt 增大仍是 0（边界层），")
print("     lambda_2 的增益应在 nt=4 或 8 就出现并平滑增长（开集效应）。")
print("     反向证伪：若 lambda_2 也要 nt>=32 才有增益，那 b51 的 +0.08..0.18 同样是网格伪影，")
print("     结论退化为'只公布 lambda_2(v1)，任何 n 都不宣传方向搜索'。")
print("  E u 方向宽度：固定 t=t*，把 u 转 delta；b51 的 u 间距是 0.20 rad，若 delta=0.2 处下降保留")
print("     仍 >0.9，则 u 分辨率从来不是问题，问题只在 t。")
