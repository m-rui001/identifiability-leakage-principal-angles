"""
b51 -- B-CLAIM-57(i): is min_v lambda_2(v) worth anything, once lambda_2 at ONE direction already
removes 0.26-0.9998 of cf1's gap?  b50 settled the moment knob (at fixed v the truncation is <= 2e-3 of
the gap even though rho_v carries 2-7 atoms), and b48/b49 settled the direction knob for the n=1 rule
(the family minimum has no closed form and one ray step buys <= 0.72).  Nobody has measured the two
knobs TOGETHER, which is the only question left that can change the route: if the certified sequence
lambda_min(G) <= min_net lambda_2 <= cf1 already sits at a few per-mille of cf1's gap, the note has a
computable two-parameter scheme instead of a formula, and the open item (1) of section 7 shrinks.

THE CONSTRUCTION (derived before any number is produced).
  For a unit v put m_k(v) = v' K' Lambda^-(k+1) K v, k = 0..3 -- FOUR fixed symmetric matrices
  M_k = K' Lambda^-(k+1) K read off once per draw, so a direction costs four quadratic forms.
  With a0 = m1/m0, b1 = m2/m0 - a0^2, a1 = (m3 - 2 a0 m2 + a0^2 m1)/(m0 b1), s1 = a0+a1, p2 = a0 a1 - b1,
  the 2-point Gauss rule is G_2^v(l) = [m0 - (m0 s1 - m1) l] / [1 - s1 l + p2 l^2] and lambda_2(v) is its
  first crossing with 1-l, i.e. the smallest non-negative root of
      p2 l^3 - (p2+s1) l^2 + [(s1+1) - (m0 s1 - m1)] l + (m0-1) = 0,
  solved here as the spectrum of the companion matrix (batched) rather than by Cardano: the discriminant
  would force a branch, and a branch is not more closed-form than "smallest non-negative root".
  Degenerate case: b1 <= 0 means rho_v has < 2 atoms, the 1-point rule is already EXACT for phi_v, and I
  set lambda_2(v) := lambda_1(v) (the family root of eq:(20)) and count the draw.
  Directions: the polar net v = cos t v1 + sin t u over u in a net of the great circle orthogonal to v1
  (b49's perp_net, re-implemented here), with t = 0 EXACTLY on the list -- b49's (E) test failed to be a
  superset of its own ray and that gap had to be reported; this time the superset is by construction.

PRE-RUN AUDIT (written after drafting the file, BEFORE its first run; the defects are recorded because
a silent fix would leave no trace of how a wrong number could have arisen -- memory rule 9):
  1. `setup` built only THREE moment matrices (range(2,4)) while `lam2_batch` indexes m0..m3, so the
     fourth moment m3 = v'K'Lambda^-4 K v was missing and a4/a1 would have taken the value of a
     lower-order matrix with no error raised. Fixed to range(2,5); the fix is checked by test V below
     (m_k against b50's per-atom formula sum_i (k_i.v)^2 / l_i^(k+1), which is an INDEPENDENT route).
  2. `perp_net` for q = 2 returned cos(th)*b + sin(th)*b, i.e. directions with norm |cos+sin| and no
     circle to sample (v1^perp is one-dimensional). Fixed to the two signed unit vectors, and the
     effective nu is reported rather than the requested one.
  3. `net_dirs` was called a second time inside a print statement to display the direction count. For
     q >= 4 that call draws from the net generator, so the displayed number came from a net that was
     never used, and the RNG stream of the second cell depended on the first. Replaced by nu*(nt+1).
  4. `lam2_batch` had a dead `np.nanmin(np.stack(...))` line immediately before the real `np.min(...)`
     over axis 1; the dead line was also wrong (nanmin of an all-NaN slice is +inf, not NaN), so it
     could not be used as a fallback even if I had wanted one. Removed.
  5. TEST E AS ORIGINALLY PRE-REGISTERED read: "min_net lambda_2 must decrease with density (a min
     over a superset) -- report the count where it does not, since that is only possible if my root
     filter is unstable across directions". The premise is false: the t-lists linspace(1e-3, pi/2, 64)
     and linspace(1e-3, pi/2, 128) do NOT nest (only the q=3 u-angle grids do), so the fine net is not
     a superset and an increase is expected behaviour, not instability. Corrected expectation: report
     the count of fine-minus-coarse > 0 and treat it as a measure of how far the coarse net is from
     saturated, with the superset claim made ONLY for t = 0 (test B), which is exact by construction.
  V  VERIFICATION, not a theorem test: on the first usable draw of the first cell, print max_k
     |m_k(v1) - sum_j (K v1)_j^2/l_j^(k+1)| / |m_k| from the atoms of Lambda, an INDEPENDENT route.
     Zero (to ~1e-14) is what licenses every number below; a rule's coefficients are only as good as
     its moments.  The check was first written as K' v, confusing v in R^q with y = K v in R^p -- the
     shape mismatch (3 vs 6) raised it, and the same confusion inside `setup` would have raised
     nothing, which is why the verification is run before the tests rather than alongside them.

PRE-REGISTERED, with the falsifier written as a number:
  A  SIDE.  min over the net of lambda_2 must be >= lam_min(G) up to the per-draw floor.  Each member is
     certified by prop:gauss, so a violation can only mean my root selection picked a non-crossing root
     of the cubic (the cubic has up to 3 real roots).  FALSIFIER of the implementation, not of the
     theorem: any violation => check the root filter, and until it is clean nothing here is evidence.
  B  SUPERSET.  min_net lambda_2 <= lambda_2(v1), because t = 0 is in the net.  Violation => the net is
     mis-assembled (the mistake I made in b49's test E).
  C  DOMINANCE OF THE DIRECTION KNOB ONCE THE MOMENT IS FREE -- the route decision, fixed now.
     gapfrac(x) := (cf1 - x)/(cf1 - lam_min).  Predictions:
        gapfrac(min_net lambda_2) >= 0.99  median, and >= 0.90 in every draw.
     FALSIFIER of "invest in the joint scheme": median < 0.90.  Then min over directions of lambda_2
     buys nothing over lambda_2(v1), and I stop on the direction leg for the second time and for the
     same reason -- measured marginal, not lack of technique.
  D  WHICH KNOB IS STRONGER.  Compare gapfrac(min_net lambda_1) [b48/b49's object: direction, n=1]
     against gapfrac(lambda_2(v1)) [moment, fixed direction] and gapfrac(min_net lambda_2).  If
     D shows the moment knob alone is the better buy, the honest recommendation is: publish the n=2
     bound at v1 and do NOT advertise a direction search.
  E  DENSITY.  Two nets (32x64 and 64x128) per draw.  min_net lambda_2 must decrease with density
     (a min over a superset) -- report the count where it does not, since that is only possible if my
     root filter is unstable across directions, and say which of the two nets the claim is made on.
     Label every search SAMPLED: the sphere is not resolved by any of these nets, and lam* = lam_min(G)
     is attained only in the limit, so the numbers below are "no better member found", never "minimum".
"""
import time

import numpy as np

rng = np.random.default_rng(60311)
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
    """same construction as b48/b49/b50 (verbatim), plus the four matrices M_k the moments need."""
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = float(np.linalg.eigvalsh(Gr)[0])
    ev, V = np.linalg.eigh(Lam)
    Li = V @ np.diag(1.0 / ev) @ V.T
    Mm = [K.T @ Li @ K]
    P = Li
    for k in range(2, 5):
        P = P @ Li
        Mm.append(K.T @ P @ K)                 # Mm[k-1] = K' Lambda^-k K, k = 1..4
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
    lbar = min(float(ev[0]), 1.0)
    if cf1 >= lbar:
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, cf1=cf1, v1=v1, m0=E0.shape[1], M=Mm,
                ev=ev, W=V, Lam=Lam, K=K, floor=100.0 * EPS * np.linalg.norm(Gr, 2))


def perp_net(v1, nu):
    """unit directions spanning v1^perp: q=2 -> the two signed basis vectors (v1^perp is 1-dimensional,
    there is no circle to sample, and the effective nu is reported), q=3 -> an angle grid on the circle,
    q>=4 -> Gaussian rows projected off v1 (b49's helper, re-derived; the audit above records why the
    q=2 branch is not the cos/sin parametrisation I first wrote)."""
    q = len(v1)
    U, _, _ = np.linalg.svd(np.eye(q) - np.outer(v1, v1))
    Bv = U[:, :q - 1]
    if q == 2:
        return np.stack([Bv[:, 0], -Bv[:, 0]])
    if q == 3:
        th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        return np.stack([np.cos(t) * Bv[:, 0] + np.sin(t) * Bv[:, 1] for t in th])
    rows = []
    while len(rows) < nu:
        g = NET.normal(size=q)
        g -= v1 * (v1 @ g)
        n = np.linalg.norm(g)
        if n > 1e-6:
            rows.append(g / n)
    return np.stack(rows)


def net_dirs(v1, nu, nt):
    """the polar net cos t v1 + sin t u, WITH t = 0 exactly (test B depends on it)."""
    U = perp_net(v1, nu)
    ts = np.concatenate(([0.0], np.linspace(1e-3, 0.5 * np.pi, nt)))
    V = (np.cos(ts)[:, None, None] * v1[None, None, :] +
         np.sin(ts)[:, None, None] * U[None, :, :]).reshape(-1, len(v1))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V, float(np.max(np.abs(np.linalg.norm(V, axis=1) - 1.0)))


def lam2_batch(VV, M, ret_degen=False):
    """lambda_2 for every direction: four quadratic forms -> cubic coefficients -> companion spectrum."""
    m = np.stack([np.einsum("ij,jk,ik->i", VV, Mk, VV) for Mk in M])       # 4 x nv
    m0, m1, m2, m3 = m
    a0 = m1 / m0
    b1 = m2 / m0 - a0 * a0
    degen = ~(b1 > 0.0)
    b1s = np.where(degen, 1.0, b1)
    a1 = (m3 - 2.0 * a0 * m2 + a0 * a0 * m1) / (m0 * b1s)
    s1 = a0 + a1
    p2 = a0 * a1 - b1s
    c3 = np.where(degen, 1.0, p2)              # keep the companion finite where b1 <= 0; those rows
    c2 = -(p2 + s1)                            # are masked as degenerate below regardless
    c1 = (s1 + 1.0) - (m0 * s1 - m1)
    c0 = m0 - 1.0
    C = np.zeros((len(VV), 3, 3))
    C[:, 0, 2] = -c0 / c3
    C[:, 1, 0] = 1.0
    C[:, 2, 1] = 1.0
    C[:, 0, 1] = -c1 / c3
    C[:, 0, 0] = -c2 / c3
    rts = np.linalg.eigvals(C)
    ok = (np.abs(rts.imag) < 1e-8) & (rts.real >= 0.0)
    rts = np.where(ok, rts.real, np.nan)
    out = np.min(np.where(np.isnan(rts), np.inf, rts), axis=1)
    # a root where the denominator is <= 0 is not a crossing of G_2 (it came in through a pole)
    den = 1.0 - s1 * out + p2 * out * out
    gval = (m0 - (m0 * s1 - m1) * out) - (1.0 - out) * den
    bad = ~(den > 0.0) | (np.abs(gval) > 1e-6 * np.maximum(1.0, m0)) | ~np.isfinite(out)
    out = np.where(degen | bad, np.nan, out)
    # the degen rows are the ones where rho_v has < 2 atoms: the 1-point rule is EXACT for phi_v there,
    # so lambda_2(v) := lambda_1(v) as promised in the docstring (a NaN would silently cost gapfrac).
    if degen.any():
        out = np.where(degen, crossings(m0, m1), out)
    if ret_degen:
        return out, degen, int(bad.sum())
    return out


def lam1_batch(VV, M):
    """the n=1 member: the family root eq:(20) of chapter 3, B = m0, C = m1."""
    return crossings(np.einsum("ij,jk,ik->i", VV, M[0], VV), np.einsum("ij,jk,ik->i", VV, M[1], VV))


def run_cell(tag, gen, n=N, dens=((32, 64), (64, 128)), verify=False):
    rows, tsec, nbad = [], 0.0, 0
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        v1, M, cf1, true, floor = s["v1"], s["M"], s["cf1"], s["true"], s["floor"]
        if verify and not rows:
            # V: m_k from the atoms of Lambda, an INDEPENDENT route to the running product Mm[k-1].
            # v lives in R^q, y = K v lives in R^p, and Lambda^-s = W diag(l^-s) W', so the ATOMIC form
            # is sum_j (W'y)_j^2 / l_j^(k+1) -- the components must be taken in the eigenbasis of Lambda.
            # Two shape/label mistakes preceded this line (K' v instead of K v, and y instead of W'y);
            # both were caught by the check printing 5.2e-01 rather than by an exception, which is the
            # point of running it before the tests.  A third draft of the atomic form, sum_i (k_i.v)^2,
            # is the same number with k_i the columns of K; I keep the eigenbasis version because it is
            # the one that cannot be confused with the running product.
            yy = s["W"].T @ (s["K"] @ v1)
            mm_at = np.array([np.sum(yy * yy / s["ev"] ** (k + 1)) for k in range(4)])
            # and a third route with NO eigen-decomposition at all: repeated linear solves, whose product
            # structure is different in kind from the running matrix product inside `setup`.
            z = s["K"] @ v1
            mm_sl = []
            for k in range(4):
                z = np.linalg.solve(s["Lam"], z)
                mm_sl.append(float(z @ (s["K"] @ v1)))
            mm_sl = np.array(mm_sl)
            mm_mx = np.array([v1 @ Mk @ v1 for Mk in M])
            sc = np.maximum(np.abs(mm_mx), 1e-300)
            print(f"{'':>28}  [V] moments m_k(v1), running product vs atoms vs repeated solves:"
                  f" max rel dev = {np.max(np.abs(mm_mx - mm_at) / sc):.2e} /"
                  f" {np.max(np.abs(mm_mx - mm_sl) / sc):.2e}"
                  f"   m(v1) = {np.array2string(mm_mx, precision=8)}")
            # and that the 1-point member really is b50's family root at v1
            print(f"{'':>28}  [V] |lambda_1(v1) - cf1| = "
                  f"{abs(float(crossings(np.array([mm_at[0]]), np.array([mm_at[1]]))[0]) - cf1):.2e}"
                  f"   (both routes must agree before anything below is evidence)")
            verify = False
        r = dict(cf1=cf1, true=true, floor=floor, gap=cf1 - true, m0=s["m0"])
        r["l2v1"], deg1, nb = lam2_batch(v1[None, :], M, ret_degen=True)
        nbad += nb
        r["l2v1"] = float(r["l2v1"][0])
        r["degen"] = bool(deg1[0])
        r["l1v1"] = float(lam1_batch(v1[None, :], M)[0])
        r["best"] = []
        for (nu, nt) in dens:
            t0 = time.time()
            VV, ue = net_dirs(v1, nu, nt)
            l2n = lam2_batch(VV, M)
            l1n = lam1_batch(VV, M)
            tsec += time.time() - t0
            r["best"].append(dict(nu=nu, nt=nt, nv=len(VV), ue=ue,
                                  min2=float(np.nanmin(l2n)), min1=float(np.min(l1n)),
                                  nan2=int(np.isnan(l2n).sum()),
                                  # the single-direction member must be inside the net (t = 0)
                                  contains_v1=float(np.nanmin(l2n)) <= r["l2v1"] + 1e-12))
        rows.append(r)
    if not rows:
        print(f"   {tag:>26}: no usable draw")
        return
    gap = np.array([r["gap"] for r in rows])
    def gf(x):
        return np.array([(r["cf1"] - (x[i] if np.isfinite(x[i]) else r["cf1"])) / r["gap"]
                         for i, r in enumerate(rows) if r["gap"] > r["floor"]])
    def idx(rows_key):
        return np.array([r[rows_key] for r in rows])
    l2v1 = idx("l2v1")
    print(f"   {tag:>26}: n={len(rows):3d} dimE0={sorted({r['m0'] for r in rows})}"
          f" degen(rho_v1 has <2 atoms, lambda_2:=lambda_1)={int(sum(r['degen'] for r in rows))}"
          f"   [A] SIDE viol: v1:{int(np.nansum(l2v1 < idx('true') - idx('floor')))}"
          f"   [B] superset fails in {sum(int(not b['contains_v1']) for r in rows for b in r['best'])}"
          f"   max||v||-1|={max(b['ue'] for r in rows for b in r['best']):.1e}"
          f"   root-filter rejections: v1 {nbad} / net {sum(b['nan2'] for r in rows for b in r['best'])}")
    for key, lab in (("l1v1", "n=1 at v1 (=cf1)"), ("l2v1", "n=2 at v1"),):
        v = gf(np.array([r[key] for r in rows]))
        print(f"{'':>28}  [{lab:>16}] gapfrac med={np.median(v):7.4f} min={np.min(v):7.4f}"
              f" max={np.max(v):7.4f}")
    for j, (nu, nt) in enumerate(dens):
        v1n = gf(np.array([r["best"][j]["min1"] for r in rows]))
        v2n = gf(np.array([r["best"][j]["min2"] for r in rows]))
        # the direction count is the ACTUAL one (nv is stored per draw): nu*(nt+1) would be a lie for
        # q=2, where v1^perp is 1-dimensional and the net carries 2 signed u regardless of nu.
        nvs = sorted({b["nv"] for r in rows for b in r["best"] if b["nu"] == nu})
        print(f"{'':>28}  [net {nu}u x {nt}t, dirs={nvs}]"
              f"   min_nets lambda_1: med={np.median(v1n):7.4f} min={np.min(v1n):7.4f}"
              f"   min_nets lambda_2: med={np.median(v2n):7.4f} min={np.min(v2n):7.4f}"
              f"   [A] viol={int(np.nansum(np.array([r['best'][j]['min2'] for r in rows]) < np.array([r['true'] for r in rows]) - np.array([r['floor'] for r in rows])))}")
    d0 = np.array([r["best"][1]["min2"] - r["best"][0]["min2"] for r in rows])
    print(f"{'':>28}  [E] fine minus coarse net: {int(np.sum(d0 > 1e-12))} draws WORSE"
          f" (expected NONZERO -- the two t-lists do not nest, see audit item 5; this is a measure of"
          f" how unsaturated the coarse net is, not a stability test)"
          f"   |gain| med = {np.median(np.abs(d0)) / max(np.median(gap), 1e-300):.4f} of the gap"
          f"   residual (1-gapfrac) med on fine net = {1 - np.median(gf(np.array([r['best'][1]['min2'] for r in rows]))):.4f}"
          f"   {tsec / max(len(rows), 1):.2f} s/draw")


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
print("[A-E] min over a polar net of the explicit-cubic lambda_2(v), against lambda_2(v1) and")
print("      min_net lambda_1 (b48/b49's object).  Route question C: gapfrac(min_net lambda_2) >= 0.99")
print("      median, >= 0.90 every draw, else the direction knob is dead even with the moment free.")
for i, (name, cs) in enumerate(SPECTRA.items()):
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)),
             verify=(i == 0))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 124)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  A 侧面：min_net lambda_2 与 lambda_2(v1) 都不得低于 lam_min(G)（超 floor 即违反）。")
print("     违反 => 我的三次方程根筛选挑了'穿过极点'的假根，先修实现再谈证据。")
print("  B 包含：网里含 t=0，故 min_net lambda_2 <= lambda_2(v1)；否则 net 组装错（b49 的 E 教训）。")
print("  C 路线判据：gapfrac(min_net lambda_2) 中位数 >= 0.99 且每格 min >= 0.90。")
print("     中位数 < 0.90 => 方向旋钮在矩自由之后仍然不买任何东西，方向腿第二次关闭。")
print("  D 哪个旋钮更强：min_net lambda_1（方向，n=1） vs lambda_2(v1)（矩，固定方向）。")
print("     若后者更大，结论是'公布 n=2 at v1，不宣传方向搜索'。")
print("  E 密度：细网 vs 粗网不是超集关系（t 列嵌套不成立，见 audit 5），故'细网更差'的样本数只反映粗网")
print("     未饱和的程度，不是稳定性判据；超集断言只对 t=0（测试 B）成立，那是构造级的。")
print("     所有搜索标为 SAMPLED：网不解析球面，'min' 只是'未找到更好成员'，方向数按实际存储的 nv 报。")
print("  V 前置校验（非定理测试）：m_k(v1) 由 Lambda 原子的闭式 sum (k_i.v1)^2/l_i^(k+1) 独立算一遍，")
print("     与递推乘积 Mm 的对偶形式比到 ~1e-14，且 lambda_1(v1) 与 cf1 同值。不同则下面全不作数。")
