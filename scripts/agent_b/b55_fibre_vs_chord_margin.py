"""
b55 -- the analytic leg promised by [B4]: the direction question for the n=1 member is a TWO-KNOB
monotone problem, and its exact answer is a closed-form margin.  Derived before running, from
eq:(family) alone, in the intercept/slope coordinates of prop:chord's own proof.

Write, for a unit v with B:=B(v) in (0,1) and t:=t(v):=C(v)/B(v) (= the mean a0 of the measure rho_v),
      r(B,t) := the smaller root of  (1-x)(1-t x) = B      <=>  t x^2 - (1+t) x + (1-B) = 0,
which is eq:(family) divided by C(v): same root, because C = B t.  The smaller root is left of the
vertex (1+t)/(2t) of the upward-opening quadratic, so implicit differentiation gives BOTH partials with
a fixed sign, no side condition beyond 0<B<1:
      dr/dt = t-part:   r(1-r)/(2tr-1-t) < 0,        dr/dB = 1/(2tr-1-t) < 0.        (mono)
So cf_1 = r(c^2, t_1) is NOT a maximum over the family: r decreases in each knob, and the member at any
other direction wins or loses according to which knob it gives up.  Let
      mu := cf_1,   d := c^2 - B(v_*) >= 0,   t_* := C(v_*)/B(v_*)
(the defect d is the quantity [B3] started printing).  Since r(B,t) <= mu, for mu at or left of the
vertex, is equivalent to (1-mu)(1-t mu) <= B, and mu itself satisfies (1-mu)(1-t_1 mu) = c^2, the two
inequalities divide:
      lambda_1(v_*) <= cf_1   <=>   t_*  >=  t_1 + (d/c^2) (1 - t_1 mu)/mu    =:  t_1 + corr(d).  (main)
That is an EXACT criterion, with the correction term written out, and its hypotheses are three numbers
I can and will check on every draw (mu <= (1+t_*)/(2t_*); 1 - t_1 mu > 0; B(v_*) < 1).

CONSEQUENCES, each of which is a test below (and this is what retires the "degenerate cell" open item):
  (a) d = 0 forces lambda_1(v_*) >= cf_1, with equality iff t_* = t_1.  In the equal-angle cells
      K = Lam^{1/2} M with M = Q1 diag(c,...,c) Q2^T, so A0 = K^T Lam^{-1} K = M^T M = c^2 I_q EXACTLY:
      B(v) = c^2 for EVERY unit v (d = 0 identically), E_0 = R^q, and t_1 = max_v C(v)/c^2, hence
      lambda_1(v_*) > cf_1 unless v_* maximises C -- and min over the WHOLE sphere of lambda_1 = cf_1,
      attained exactly at the Rayleigh maximisers of B2.  That is a provable statement in those cells
      and it is the one place where the direction question has a closed answer.
  (b) lambda_1(v_*) < cf_1 REQUIRES d > 0: leaving E_0 costs intercept and buys slope, and (main) says
      the slope gain must exceed corr(d) = (d/c^2)(1-t_1 mu)/mu.  So the 280 dim E_0 = 1 draws where
      [B3] saw cf_1 > lambda_1(v_*) are not "v_* is a better direction" but "v_* trades B for t".

PRE-REGISTERED, falsifiers as numbers (memory rule 8: signs are registered only where I derived the
derivative, magnitudes only as intervals):
  T0  COORDINATE GATE (defining identity of my own rewrite, checked FIRST -- rule 9): on 4000 random
      (B,t), |r(B,t) - crossings(B, B t)| <= 1e-13 and r(c^2,t) == cf_of(t,c^2).  FALSIFIER: any draw
      above 1e-10 => I mis-read eq:(family), and every line below is void (this is exactly the failure
      mode that produced retraction (p)).
  T1  MONOTONITY, tested where the search is EXHAUSTIVE (rule 3): r(0.5, t) strictly decreasing on a
      20000-point t-sweep over [1e-6, 1e6]; r(B, 3.0) strictly decreasing on a 20000-point B-sweep over
      (0,1); central differences match the closed forms of (mono) to 1e-7 relative on 2000 interior
      pairs.  FALSIFIER: one violation kills (mono); a derivative mismatch kills the closed forms only.
  T2  THE EXACT CRITERION on b53's SAME 320 draws (same seed and call order, eight cells): per draw,
      sign(lambda_1(v_*) - cf_1) must be OPPOSITE to sign(T) with T := t_* - t_1 - corr(d); and the
      count of T < 0 must equal the count of cf_1 > lambda_1(v_*) that [B3] printed (40/40 in the
      equal-angle cell, 0/280 elsewhere).  FALSIFIER: any mismatch => my hand division of the two
      inequalities is wrong, and with it [B4] AND the explanation I wrote into Remark rem:tight.
  T3  HYPOTHESES of (main) per draw: mu <= (1+t_*)/(2t_*), 1 - t_1 mu > 0, B(v_*) < 1, d >= -floor.
      Any failure is reported and that draw is EXCLUDED from T2's claim, not silently averaged in.
  T4  THE EQUAL-ANGLE COROLLARY, exhaustively: on new q=2 and q=3 equal-angle cells, max over a
      20000-point net and 20000 random unit vectors of C(v) equals lambda_max(B2), B(v) == c^2 to
      <= 1e-14 everywhere, and min over the net of lambda_1(v) equals cf_1 with 0 competitors below
      cf_1 - floor; while lambda_1(v_*) > cf_1 on every draw.  FALSIFIER: a single grid point with
      lambda_1 < cf_1 - floor would mean r is not monotone in t, i.e. T1's sweep missed it, and (a) dies.
  T5  B-CLAIM-58(i)'s new-table leg: NEW seed, p doubled (p=12, q=3) and q doubled (p=7, q=6) and a
      non-equal-angle 4-column spread -- re-run T2 there, and report gapfrac(lambda_2(v_ray)) with the
      registered falsifier "< 0.90 on a new family => the 0.99 was a coincidence of b53's tables".
      No sign registered for the size (the remainder of the n=2 rule is still unbounded for me).
"""
import numpy as np

rng = np.random.default_rng(60311)          # b53's seed and call order: the SAME 320 draws
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
    c3 = p2
    C = np.zeros((len(VV), 3, 3))
    C[:, 0, 2] = -(m0 - 1.0) / c3
    C[:, 0, 1] = -((s1 + 1.0) - (m0 * s1 - m1)) / c3
    C[:, 0, 0] = -(-(p2 + s1)) / c3
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


def setup(Lam, K):
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    w, W = np.linalg.eigh(Gr)
    true = float(w[0])
    y = W[p:, 0]                                  # q-BLOCK of the minimal eigenvector (b53's assert kept)
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
    return dict(true=true, A0=A0, B2=B2, c2=c2, t1=t1, cf1=cf1, v1=v1, vstar=vstar, y=y,
                lam_minL=float(ev[0]), M=Mm, K=K, Lam=Lam, floor=100.0 * EPS * np.linalg.norm(Gr, 2),
                m0=E0.shape[1])


def ray(s):
    """b49/b50's two-jet ray, verbatim."""
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
    tp = -L1 / (2.0 * L2)
    return np.cos(tp) * v1 + np.sin(tp) * u, tp


# ---------------------------------------------------------------- the new mathematics, written from the derivation
def r_bt(B, t):
    """Smaller root of (1-x)(1-t x)=B in the INTERCEPT/SLOPE coordinates (B,t).  Independent of
    crossings(), which uses (B,C): T0 exists to prove the two are the same root."""
    B = np.asarray(B, dtype=float)
    t = np.asarray(t, dtype=float)
    disc = (1.0 - t) ** 2 + 4.0 * t * B
    return ((1.0 + t) - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * t)


def corr_term(d, c2, t1, mu):
    """corr(d) = (d/c^2)(1 - t_1 mu)/mu, the slope a direction may lose and still tie cf_1."""
    return (d / c2) * (1.0 - t1 * mu) / mu


# ------------------------------------------------------------------ T0: the coordinate gate, run FIRST
print("=" * 124)
print("[T0] r(B,t) must be the SAME root as crossings(B, B t), and r(c2,t) the same as cf_of(t,c2):")
print("     my rewrite of eq:(family) is otherwise just a new name for an old formula.")
g = np.random.default_rng(4242)
Bs = g.uniform(1e-4, 1.0 - 1e-6, 4000)
ts = 10.0 ** g.uniform(-3.0, 3.0, 4000)
a_r, a_c = r_bt(Bs, ts), crossings(Bs, Bs * ts)
d_c2 = np.array([abs(r_bt(c2, tt) - cf_of(tt, c2)) for c2, tt in
                 zip(g.uniform(1e-4, 0.9999, 4000), 10.0 ** g.uniform(-3, 3, 4000))])
scale = np.maximum(np.abs(a_c), 1e-3)
print(f"     max |r_bt - crossings| = {np.max(np.abs(a_r - a_c)):.3e}   (as a fraction of the root:"
      f" {np.max(np.abs(a_r - a_c) / scale):.3e})   max |r_bt(c2,t) - cf_of(t,c2)| = {np.max(d_c2):.3e}")
# DISCRIMINANT CHECK, CORRECTED.  My first draft compared eq:(family)'s discriminant with (1-t)^2+4tB
# and printed 9.7e+05: the two quantities differ by the factor B^2 (D_(20) = B^2 * D_(B,t)), so that line
# was comparing two things that are not equal, exactly the label-vs-expression defect of retraction (p).
# The identity as derived is D_(20) = B^2 [(1-t)^2 + 4tB], and that is what is now tested.
disc_20 = (Bs + Bs * ts) ** 2 - 4.0 * (Bs * ts) * Bs * (1.0 - Bs)
disc_bt = (1.0 - ts) ** 2 + 4.0 * ts * Bs
print(f"     DISCRIMINANT: D_(20) vs B^2*(1-t)^2+4tB): max |ratio-1| ="
      f" {np.max(np.abs(disc_20 / (Bs ** 2 * disc_bt) - 1.0)):.3e}   (the raw diff, which my first draft"
      f" printed, is max {np.max(np.abs(disc_20 - disc_bt)):.3e} -- NOT an identity, my check was wrong)")

# ------------------------------------------------------------------------------ T1: monotone, exhaustively
print("\n" + "=" * 124)
print("[T1] both partials of r at the smaller root are negative: dr/dt = r(1-r)/(2tr-1-t),")
print("     dr/dB = 1/(2tr-1-t), and 2tr-1-t<0 because the smaller root is left of the vertex (1+t)/2t.")
print("     Sweeps are 20000-point and cover the whole branch, so this leg is exhaustive in 1-D.")
tt = np.geomspace(1e-6, 1e6, 20000)
r_t = r_bt(0.5, tt)
viol_t = int(np.sum(np.diff(r_t) > 0.0))
BB = np.linspace(1e-9, 1.0 - 1e-9, 20000)
r_B = r_bt(BB, 3.0)
viol_B = int(np.sum(np.diff(r_B) > 0.0))
pairs = list(zip(g.uniform(1e-3, 0.99, 2000), 10.0 ** g.uniform(-2, 2, 2000)))
err_t, err_B, vm, worst = [], [], 0, (0.0, None)
# STEP SIZE, CORRECTED.  My first draft used an ABSOLUTE h=1e-7 for both knobs and got
# max rel err dr/dt = 7.0e-05 (vs 5.9e-08 for dr/dB): at t ~ 1e2 the derivative is itself O(r/t), so a
# fixed 1e-7 step makes the difference quotient round-off dominated -- a defect of the check, not of the
# closed form.  The step is now RELATIVE to each knob, which is the scaling the derivative lives on.
for B0, t0 in pairs:
    r0 = float(r_bt(B0, t0))
    vtx = (1.0 + t0) / (2.0 * t0)
    if r0 >= vtx:
        vm += 1
        continue
    ht, hb = 1e-6 * t0, 1e-6 * B0
    fd_t = (float(r_bt(B0, t0 + ht)) - float(r_bt(B0, t0 - ht))) / (2 * ht)
    fd_B = (float(r_bt(B0 + hb, t0)) - float(r_bt(B0 - hb, t0))) / (2 * hb)
    den = 2.0 * t0 * r0 - 1.0 - t0
    cl_t, cl_B = r0 * (1.0 - r0) / den, 1.0 / den
    e_t = abs(fd_t - cl_t) / abs(cl_t)
    err_t.append(e_t)
    err_B.append(abs(fd_B - cl_B) / abs(cl_B))
    if e_t > worst[0]:
        worst = (e_t, (B0, t0, r0, cl_t, fd_t))
print(f"     t-sweep at B=0.5: increases on {viol_t}/19999 steps, range r = {r_t.min():.3e}..{r_t.max():.6f}")
print(f"     B-sweep at t=3.0: increases on {viol_B}/19999 steps, range r = {r_B.min():.6f}..{r_B.max():.6f}")
print(f"     finite-difference vs closed form (relative steps): max rel err dr/dt={max(err_t):.2e}"
      f" dr/dB={max(err_B):.2e}   median dr/dt err={np.median(err_t):.2e}")
print(f"     worst dr/dt pair: B={worst[1][0]:.4f} t={worst[1][1]:.4f} r={worst[1][2]:.6f}"
      f" closed={worst[1][3]:.3e} fd={worst[1][4]:.3e}"
      f"   (2tr-1-t >= 0, i.e. root at/right of vertex, in {vm}/{len(pairs)} pairs"
      f" -- those are the LARGER root by construction of r_bt, so they are excluded, not averaged)")

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


# ---------------------------------------------------------------------------- T2/T3: the criterion on b53's draws
def crit(s, vs):
    """Everything (main) needs, plus its three hypotheses, for one draw."""
    c2, t1, mu = s["c2"], s["t1"], s["cf1"]
    Bst = float(vs @ s["A0"] @ vs)
    Cst = float(vs @ s["B2"] @ vs)
    tst = Cst / Bst
    d = c2 - Bst
    T = tst - t1 - corr_term(d, c2, t1, mu)
    l1 = float(crossings(np.array([Bst]), np.array([Cst]))[0])
    # [T2b] ADDED AFTER THE FIRST RUN, and it is where the algebra pays: linearising r at (c^2,t_1),
    #     lambda_1(v_*) - cf_1 = (dr/dt)(t_*-t_1) + (dr/dB)(B_*-c^2), and because
    #     dr/dt = mu(1-mu) dr/dB and (1-mu)(1-t_1 mu) = c^2, the two coefficients agree on the
    #     intercept term with corr(d) EXACTLY: (dr/dt) * corr(d) = d.  So the WHOLE two-knob first
    #     order collapses onto the single margin T:   lambda_1(v_*) - cf_1 ~ [mu(1-mu)/(2t_1 mu-1-t_1)] T.
    #     That is the quantitative form of (main); the sign statement was only its shadow.
    den = 2.0 * t1 * mu - 1.0 - t1
    hyp = dict(vtx=mu <= (1.0 + tst) / (2.0 * tst), pos=1.0 - t1 * mu > 0.0, Blt1=Bst < 1.0,
               dneg=d >= -s["floor"])
    return dict(l1=l1, cf1=mu, T=T, d=d, tstar=tst, t1=t1, gap=mu - s["true"], true=s["true"],
                floor=s["floor"], m0=s["m0"], l2=float(lam2_batch(vs[None, :], s["M"])[0]),
                lin=(l1 - mu) / (mu * (1.0 - mu) / den * T) if T != 0.0 else np.nan,
                rbt=float(r_bt(Bst, tst)), **hyp)


def run_cell(tag, gen, n=N, extra=None):
    rows, skipped = [], 0
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            skipped += 1
            continue
        if not (s["cf1"] - s["true"] > s["floor"]):
            skipped += 1
            continue
        c = crit(s, s["vstar"])
        rr = ray(s)
        c["l2ray"] = np.nan
        if rr is not None:
            vr, _ = rr
            c["l2ray"] = float(lam2_batch(vr[None, :], s["M"])[0])
            c["gf_ray"] = (s["cf1"] - c["l2ray"]) / c["gap"]
        rows.append(c)
        if extra is not None:
            extra(s)
    if not rows:
        print(f"   {tag:>26}: no usable draw ({skipped} skipped)")
        return None
    a = lambda k: np.array([r[k] for r in rows])
    ok_hyp = np.all([r["vtx"] and r["pos"] and r["Blt1"] and r["dneg"] for r in rows])
    # T2 as registered: sign(l1 - cf1) opposite to sign(T), on the draws whose hypotheses hold
    sel = np.array([r["vtx"] and r["pos"] and r["Blt1"] and r["dneg"] for r in rows])
    l1, cf1, Tv = a("l1"), a("cf1"), a("T")
    mism = int(np.sum(sel & (((l1 <= cf1) & (Tv < 0.0)) | ((l1 > cf1) & (Tv >= 0.0)))))
    # the code's lambda_1(v*) must be r(B*,t*) -- T0's identity re-checked on real data
    idmax = float(np.max(np.abs(l1 - a("rbt")) / np.maximum(l1, 1e-12)))
    gf2 = (a("cf1") - a("l2")) / a("gap")
    gfr = np.array([r["gf_ray"] for r in rows if "gf_ray" in r])
    lin = a("lin")
    glfr = gfr[np.isfinite(gfr)] if gfr.size else np.array([])
    print(f"   {tag:>26}: n={len(rows):3d} skip={skipped:2d} dimE0={sorted({r['m0'] for r in rows})}"
          f"   [T3] all hypotheses: {ok_hyp}"
          f"   [T0'] max |lambda_1(v*)-r(B*,t*)|/lambda_1 = {idmax:.1e}")
    # THE COUNT RELATION, CORRECTED.  I registered "#(T<0) must equal #(cf_1 > lambda_1(v_*))", which is
    # the wrong side of the iff: (main) reads lambda_1(v_*) <= cf_1 <=> T >= 0, so #(T<0) must equal
    # #(lambda_1(v_*) > cf_1).  Both counts are printed so the mis-registration is visible, not hidden.
    print(f"{'':>28}  [T2] sign mismatches = {mism}/{int(np.sum(sel))}"
          f"   #(T<0)={int(np.sum(Tv < 0))} vs #(lambda_1(v*)>cf1)={int(np.sum(l1 > cf1))}"
          f" [#(cf1>lambda_1)={int(np.sum(cf1 > l1))}]   d=c2-B(v*): med={np.median(a('d')):.2e}"
          f" max={np.max(a('d')):.2e}   med|lambda_1(v*)-cf1|/gap={np.median(np.abs(l1 - cf1) / a('gap')):.2e}")
    print(f"{'':>28}  [T2b] first-order |lambda_1(v*)-cf1| / |mu(1-mu)T/(2t1mu-1-t1)|:"
          f" med={np.median(np.abs(lin[np.isfinite(lin)])):.4f}"
          f" min={np.min(np.abs(lin[np.isfinite(lin)])):.4f} max={np.max(np.abs(lin[np.isfinite(lin)])):.4f}"
          f"   #(>0.5 or <2)={int(np.sum((np.abs(lin) < 0.5) | (np.abs(lin) > 2.0)))}")
    if glfr.size:
        print(f"{'':>28}  [ray] gapfrac(lambda_2(v_ray)) med={np.median(glfr):.4f}"
              f" min={np.min(glfr):.4f} (#ray={glfr.size}) #<0.90={int(np.sum(glfr < 0.90))}"
              f"   gapfrac(lambda_2(v*)) med={np.median(gf2):.2e} max={np.max(gf2):.2e}")
    return rows


print("\n" + "=" * 124)
print("[T2/T3] the exact criterion on b53's SAME draws: lambda_1(v*) <= cf_1 <=> T >= 0, with")
print("        T := t_* - t_1 - (d/c^2)(1-t_1 mu)/mu, d = c^2 - B(v_*).  #(T<0) must equal")
print("        #(cf_1 > lambda_1(v_*)) cell by cell ([B3] printed 40/40 for 'equal 20d', 0 elsewhere).")
for name, cs in SPECTRA.items():
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

# ---------------------------------------------------------------------------- T4: the equal-angle corollary
print("\n" + "=" * 124)
print("[T4] equal-angle cells: A0 = K^T Lam^-1 K = M^T M = c^2 I_q EXACTLY, so B(v)=c^2 for every unit v")
print("     (d=0 identically), E_0 = R^q, and min over the sphere of lambda_1(v) = cf_1, attained at the")
print("     Rayleigh maximisers of B2.  Tested on nets dense enough that the claim is searched, not sampled:")
print("     q=2 on a 20000-point circle (exhaustive up to the 3.1e-4 rad spacing), q=3 on 20000 random +")
print("     40000 spiral points.  Falsifier: one competitor below cf_1 - floor.")
eq2 = np.cos(np.deg2rad(20.0))
for q, nm in [(2, "equal 20d, q=2"), (3, "equal 20d, q=3")]:
    badB = badC = badmin = nuse = 0
    worst = 0.0
    for _ in range(12):
        Lam, K = (lambda L: (L, blocks(L, [eq2] * q)))(lam_spd(6))
        s = setup(Lam, K)
        if s is None or not (s["cf1"] - s["true"] > s["floor"]):
            continue
        nuse += 1
        worst = max(worst, float(np.max(np.abs(s["A0"] - s["c2"] * np.eye(q)))))
        th = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
        V = (np.column_stack([np.cos(th), np.sin(th)]) if q == 2 else None)
        if q == 3:
            i = np.arange(40000) + 0.5
            phi = np.arccos(1.0 - 2.0 * i / 40000)
            th2 = (np.pi * (1.0 + np.sqrt(5.0)) * i) % (2.0 * np.pi)
            V = np.column_stack([np.sin(phi) * np.cos(th2), np.sin(phi) * np.sin(th2), np.cos(phi)])
            Z = np.random.default_rng(5).normal(size=(20000, 3))
            V = np.vstack([V, Z / np.linalg.norm(Z, axis=1, keepdims=True)])
        BV = np.einsum("ij,jk,ik->i", V, s["A0"], V)
        CV = np.einsum("ij,jk,ik->i", V, s["B2"], V)
        l1V = crossings(BV, CV)
        badB += int(np.sum(np.abs(BV - s["c2"]) > 1e-14))
        badC += int(np.sum(CV > np.linalg.eigvalsh(s["B2"])[-1] + 1e-12))
        floor = s["floor"]
        badmin += int(np.sum(l1V < s["cf1"] - floor))
        vs = s["vstar"]
        cst = crit(s, vs)
        worst = max(worst, abs(cst["d"]))
        if cst["T"] >= 0.0:
            badmin = -999
            break
    print(f"   {nm:>26}: draws={nuse:2d} max|A0-c2 I|={worst:.2e}   B(v)=c2 fails on {badB} grid pts"
          f"   C(v)>lam_max(B2) on {badC}   lambda_1(v)<cf1-floor on {badmin}"
          f"   (T>=0 at v*: {'NO' if badmin == -999 else 'never' if badmin >= 0 else '?'})")

# ------------------------------------------------------------------- T5: new tables (B-CLAIM-58(i) leg)
print("\n" + "=" * 124)
print("[T5] B-CLAIM-58(i): NEW seed, doubled p, doubled q, and a spread whose rows are not equal-angled.")
print("     Same criterion T2 must hold; the registered falsifier for the 0.99 gapfrac is a cell whose")
print("     lambda_2(v_ray) drops below 0.90 of the gap.")
rng = np.random.default_rng(77777)


def lam_spd2(p):
    E = rng.normal(size=(100, p))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E.T @ E


def blocks2(Lam, cosines):
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))
    M = Q1 @ np.diag(cosines) @ Q2.T
    ev, V = np.linalg.eigh(Lam)
    return (V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T) @ M


sp = [np.cos(np.deg2rad(a)) for a in (3, 17, 44, 61, 79, 86)]
run_cell("q=3 p=12 spread 5-75", lambda: (lambda L: (L, blocks2(L, SPECTRA["spread 5-75"])))(lam_spd2(12)),
         n=30, extra=None)
run_cell("q=3 p=12 spread 2-45-88", lambda: (lambda L: (L, blocks2(L, SPECTRA["spread 2-45-88"])))(lam_spd2(12)), n=30)
run_cell("q=6 p=7 spread 6-col", lambda: (lambda L: (L, blocks2(L, sp)))(lam_spd2(7)), n=25)
run_cell("q=3 p=12 equal 20d", lambda: (lambda L: (L, blocks2(L, [eq2] * 3)))(lam_spd2(12)), n=25)

# -------------------------------------------------------------------- T6: is the ray leg's loss monotone in q?
print("\n" + "=" * 124)
print("[T6] POST-RUN LEG (added AFTER [T5] fired, so it is exploration, not evidence for the registered")
print("     falsifier): the q=6 cell gave gapfrac(lambda_2(v_ray)) median 0.7387, min 0.5571, so the")
print("     0.99 of B-CLAIM-58 is not table-independent.  The question is now whether the loss is")
print("     MONOTONE in q, i.e. whether the analytic ray degrades as the number of columns grows.  No")
print("     sign registered (rule 8): I cannot bound the remainder of the two-jet ray in q.")
for q in (2, 3, 4, 5, 6, 8):
    csq = [np.cos(np.deg2rad(a)) for a in np.linspace(3.0, 86.0, q)]
    rng = np.random.default_rng(77777 + q)
    run_cell(f"T6 q={q} p={q+1} spread", lambda csq=csq: (lambda L: (L, blocks2(L, csq)))(lam_spd2(q + 1)),
             n=20)

# --------------------------------------------------------- T7: how far in T does the linear law stay linear?
print("\n" + "=" * 124)
print("[T7] POST-RUN, and it exists because [T2b]'s ratio was 0.9996-1.0003 on EVERY cell: a first-order")
print("     law that is exact to 4 decimals on the draws I happen to sample is not yet a law, it is a")
print("     smallness of |T|.  So I drive T large on purpose -- v(theta) = cos(th) v_* + sin(th) w with")
print("     w a unit vector orthogonal to v_* (so ||v||=1 by construction, and that is printed) -- and")
print("     measure the ratio (lambda_1(v)-cf_1) / (coef * T(v)) against |T|.  Reading registered before")
print("     running: if the 1%-band ends at |T| ~ 1e-2 the linear law is a REGIME statement and the")
print("     chapter may only carry the sign statement (main); if the band is flat out to |T| ~ 1 the")
print("     collapse is structural and I will state the closed form with its remainder.  NB the RNG has")
print("     advanced through [T5]/[T6], so these are NEW draws, not b53's -- labelled as such.")
for q, cs in [(3, SPECTRA["spread 2-45-88"]), (4, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])]:
    prof = []
    for _ in range(6):
        Lam, K = (lambda L: (L, blocks(L, cs)))(lam_spd(6 if q == 3 else 7))
        s = setup(Lam, K)
        if s is None or not (s["cf1"] - s["true"] > s["floor"]):
            continue
        vs = s["vstar"]
        W = np.eye(len(vs)) - np.outer(vs, vs)
        ew, Vw = np.linalg.eigh(W)
        w = Vw[:, np.argmax(ew)]
        if abs(float(w @ vs)) > 1e-12:
            continue
        mu, t1, c2 = s["cf1"], s["t1"], s["c2"]
        coef = mu * (1.0 - mu) / (2.0 * t1 * mu - 1.0 - t1)
        for th in np.linspace(0.0, np.pi / 2.0, 2000):
            v = np.cos(th) * vs + np.sin(th) * w
            Bv, Cv = float(v @ s["A0"] @ v), float(v @ s["B2"] @ v)
            Tv = Cv / Bv - t1 - corr_term(c2 - Bv, c2, t1, mu)
            lv = float(crossings(np.array([Bv]), np.array([Cv]))[0])
            prof.append((abs(Tv), (lv - mu) / (coef * Tv) if Tv != 0.0 else np.nan,
                         abs(np.linalg.norm(v) - 1.0)))
    if not prof:
        print(f"   q={q}: no usable draw")
        continue
    TT = np.array([x[0] for x in prof])
    RR = np.array([x[1] for x in prof])
    ok = np.isfinite(RR) & (TT > 0)
    band = ok & (np.abs(RR - 1.0) <= 0.01)
    print(f"   q={q} ({int(ok.sum())} pts, max |T|={TT[ok].max():.3e}, max ||v||-1|={max(x[2] for x in prof):.1e}):"
          f" ratio in [0.99,1.01] up to |T|<={np.max(TT[band]) if band.any() else 0:.3e}"
          f"   ratio range over all pts: [{np.min(RR[ok]):.4f},{np.max(RR[ok]):.4f}]"
          f"   #(sign flip)={int(np.sum(((RR > 0) & (TT < 0)) | ((RR < 0) & (TT > 0))))}")
    for lo in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        sel2 = ok & (TT < lo)
        if sel2.any():
            print(f"        |T| < {lo:.0e}: {int(sel2.sum()):5d} pts, max |ratio-1| ="
                  f" {np.max(np.abs(RR[sel2] - 1.0)):.2e}")

print("\n" + "=" * 124)
print("判读（预登记，跑前写）：")
print("  T0 若 |r_bt - crossings| 任一点 >1e-10，则我对 eq:(family) 的改写读错了，后面全部作废——")
print("     这正是 (p) 号撤回的失败模式，所以这条必须先跑。")
print("  T1 两个偏导都应为负（小根在顶点左边）。1D 全程扫描是穷尽的：任何一处上升即 (mono) 死，")
print("     [B4] 与 rem:tight 里那句「固定 B 时根随 C 递减」同时撤。中心差分对不上只杀闭式，不杀单调。")
print("  T2 每格 #(T<0) 必须等于 #(cf1 > lambda_1(v*))，且 sign 不一致数 = 0。尤其 equal 20d 那格")
print("     应恰好 40 个 T<0（d=0 而 t*<t1，(a) 的预言），其余七格 T>=0。若有 dimE0=1 的格出现")
print("     T<0（即 v* 用截距换斜率换输了），那不是失败——是 (main) 抓到了一个新机制，我要单独报。")
print("  T3 三个前提逐抽样打印通过与否；不通过的抽样从 T2 的计数里剔除，不混进平均。")
print("  T4 等角格里 A0 与 c^2 I 的差应在 1e-15（这是构造，不是拟合）；网上 min lambda_1 不得跌破")
print("     cf1-floor；v* 处 T 应 <0（于是 cf1 是整球面 n=1 族的严格最小，v* 反而更差）。")
print("  T5 新表上 T2 仍 0 失配则 (main) 与 b53 的表族无关；若 lambda_2(v_ray) 的 gapfrac 出现 <0.90，")
print("     B-CLAIM-58 的 0.99 按 (i) 降级为「平均更好」。两种结果都写，不挑。")
print("  登记本身的三处缺陷（跑前发现两处、跑后发现一处，都改在代码里并注明，不改登记的原话）：")
print("   (1) T0 的绝对阈值 1e-13 写在 (B,C) 形式上，而 (B,C) 的求根必然比 (B,t) 形式多一次乘除；")
print("       真正的门是相对量与 1e-10 的证伪器，两者都按原样报。")
print("   (2) 我登记的计数关系 #(T<0)=#(cf1>lambda_1) 是 (main) 的反面，正确关系是 #(T<0)=")
print("       #(lambda_1(v*)>cf1)；两个计数都印出来，让误登记可见。")
print("   (3) 判据里那条「判别式恒等」把 D_(20) 与 (1-t)^2+4tB 直接相减，差了 B^2 因子——印出 9.7e+05")
print("       就是它；现改为按 B^2 归一后比，另把原始差留着当 specimen（记忆规则 9 的推论：读表达式，")
print("       不读标签）。")
