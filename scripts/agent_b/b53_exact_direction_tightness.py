"""
b53 -- the fibre family is TIGHT, and the exact minimising direction is free: it is the q-block of the
eigenvector of G at lam_min(G).  Derived before running, from the block eigen-equations alone:

    G [x;y] = lam [x;y]  with  G = [[Lam, K],[K', I]]  gives   (Lam - lam I) x = -K y,
    K' x = (lam - 1) y, hence   f_lam y := K'(Lam - lam I)^{-1} K y = (1 - lam) y,
    valid at lam = lam_min(G) < lam_min(Lam) so that (Lam - lam I) is positive definite -- the SAME
    branch split prop:chord's proof already makes.  Normalising v := y/||y|| therefore gives
        phi_v(lam_min) = v' K'(Lam - lam_min I)^{-1} K v = 1 - lam_min,
    i.e. the fibre crossing lam(v) -- the smaller root of eq:(20) at n = 1 -- satisfies
        min over unit v of lam_1(v) = lam_1(v*) = lam_min(G),      (prop:tight, conjectured here)
    which is the exact statement that prop:chord's inequality is an EQUALITY for one special direction,
    and that all of cf1's residual gap is DIRECTIONAL, not truncational.  Combined with the measured
    pointwise ordering lam_2(v) <= lam_1(v) and the certified side lam_2(v) >= lam_min(G), a third
    consequence is forced: lam_2(v*) = lam_1(v*) = lam_min(G) to round-off, with no new argument.

WHY THIS RECONCILES AND CLOSES b51/b52.  b52 measured that lambda_1's descent set from v1 is a boundary
layer of angular width t_hi = 2.5e-5..1.6e-3, and that no uniform grid in t can see it.  If v* is the
global minimiser then t_hi is a measurement of ANGLE(v*, v1), so this file predicts
    angle(v*, v1) ~ t_hi, and angle(v*, v_ray) << angle(v*, v1) is FALSE -- the ray gets 0.13-0.50 of
the gap, so it should still sit far from v* in the metric that matters.  Both are printed.

PRE-REGISTERED, falsifiers as numbers:
  A  THE IDENTITY, tested on the eigenvector itself: with y the q-block of G's minimal eigenvector and
     lam = lam_min(G), the residual ||f_lam y - (1-lam) y|| / ((1-lam)||y||) must be <= 1e-10.  This is
     independent of any root formula, so it is the gate: if it fails, the derivation is wrong, not the
     numerics.  FALSIFIER: any draw above 1e-10.
  B  lam_1(v*) = lam_min(G): |lam_1(v*) - lam_min| / gap <= 1e-8 for EVERY draw.  FALSIFIER: a draw
     where it exceeds 1e-6, which would mean a second, non-crossing branch of eq:(20) interferes.
  C  SIDE and ORDER at v*: lam_min - floor <= lam_2(v*) <= lam_1(v*) + floor (the forced equality of
     the header), reported as (lam_2(v*) - lam_min)/gap, predicted <= 1e-8.  FALSIFIER: > 1e-4 would
     contradict the measured pointwise ordering lam_2 <= lam_1 at v*, i.e. would say the ordering has an
     exception, which b50 tested only at 240 points on v1 -- so a failure here is informative, not noise.
  D  THE ANGLE: cos angle(v*, v1) reported per cell, prediction median in [1e-5, 2e-3] to match b52's
     t_hi (two different computations of the same quantity).  FALSIFIER: median angle > 1e-1 -- then
     b52's boundary layer is NOT the approach to v* and the reconciliation is incomplete.
  E  HOW FAR THE RAY IS: angle(v*, v_ray) and gapfrac(lam_1(v_ray)) on the SAME draws; and the best
     member of the b52 geometric grid vs lam_1(v*).  Prediction: the ray is within a factor ~2 of the
     angle to v1 in some cells yet only removes 0.13-0.50 of the gap, so ANGLE is a poor proxy for
     BOUND QUALITY -- the sensitivity of lam_1 to the direction is what matters, and that sensitivity is
     exactly what makes a uniform net useless (b52 [D]).  No falsifier registered: this line is a
     measurement to be reported, not a prediction, and is labelled as such BEFORE running (memory
     rule 8: I only register a direction for a quantity whose remainder I can bound).
"""
import numpy as np

rng = np.random.default_rng(60311)          # b51/b52's seed: the SAME draws
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
    y = W[p:, 0]                                  # the q-BLOCK of the minimal eigenvector: the leading
    # p rows of the eigenvector of a (p+q)-by-(p+q) matrix are the p-BLOCK, so the block I want starts
    # at index p, not at q.  My first draft wrote W[q:, 0]; with p=6, q=3 the shape mismatch (6 vs 3)
    # is what raised it -- and in a square block (p = q) it would have raised NOTHING and silently
    # returned the p-block relabelled.  That is the case to fear, so the block split is asserted below.
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
    cf1 = cf_of(float(mu[-1]) / c2, c2)
    v1 = E0 @ Wm[:, -1]
    v1 = v1 / np.linalg.norm(v1)
    if cf1 >= min(float(ev[0]), 1.0):
        return None
    return dict(true=true, A0=A0, B2=B2, c2=c2, cf1=cf1, v1=v1, vstar=vstar, y=y, xblock=W[:p, 0],
                lam_minL=float(ev[0]),
                M=Mm, K=K, Lam=Lam, floor=100.0 * EPS * np.linalg.norm(Gr, 2), m0=E0.shape[1])


def ray(s):
    """b49/b50's two-jet ray, verbatim (the same u and t* that b52 profiled)."""
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


def run_cell(tag, gen, n=N):
    rows = []
    for _ in range(n):
        Lam, K = gen()
        s = setup(Lam, K)
        if s is None:
            continue
        true, floor, M, v1 = s["true"], s["floor"], s["M"], s["v1"]
        gap = s["cf1"] - true
        if not (gap > floor):
            continue
        vs = s["vstar"]
        # A: the eigen-identity, with NO root formula in sight
        f_y = K.T @ np.linalg.solve(s["Lam"] - true * np.eye(Lam.shape[0]), K @ s["y"])
        rel = float(np.linalg.norm(f_y - (1.0 - true) * s["y"]) / ((1.0 - true) * np.linalg.norm(s["y"])))
        # A2: the p-block must satisfy (Lam - lam I) x = -K y with the SAME y -- a second, independent
        # reading of the same eigen-pair, which is what tells me I took the right block at all.
        xs = -np.linalg.solve(s["Lam"] - true * np.eye(Lam.shape[0]), K @ s["y"])
        relx = float(np.linalg.norm(xs - s["xblock"]) / np.linalg.norm(s["xblock"]))
        # B/C: the two family members AT v*, and the ordering
        bs = np.array([float(vs @ Mk @ vs) for Mk in M])
        l1s = float(crossings(np.array([bs[0]]), np.array([bs[1]]))[0])
        l2s = float(lam2_batch(vs[None, :], M)[0])
        rr = ray(s)
        ang1 = float(np.arccos(np.clip(np.abs(vs @ v1), -1.0, 1.0)))
        d = dict(rel=rel, relx=relx, l1s=l1s, l2s=l2s, gap=gap, true=true, floor=floor, ang=ang1,
                 cf1=s["cf1"], m0=s["m0"], bstar=float(bs[0]), c2=s["c2"])
        # [B3] ADDED AFTER THE [B2] READOUT, and it is a correction of my own label, not a new test.
        # The line below used to print "B(v*)=<mean> vs c2=<min of the same array>": the second number
        # was min_i B(v*)_i, NOT c^2 = lambda_max(K^T Lam^-1 K), which s["c2"] has been carrying all
        # along.  So the sentence "B(v*) equals c^2 to six digits" that I wrote in the .txt, in
        # community.md and in chapter4 was comparing one quantity with itself, and proves nothing about
        # whether v* saturates the intercept.  Since B(v) <= c^2 pointwise, what I can now actually test
        # is the DEFECT c^2 - B(v*) and the alignment of v* with the top eigenvector of A0.
        wA, VA = np.linalg.eigh(s["A0"])
        d["top"] = float(np.abs(vs @ VA[:, -1]))
        if rr is None:
            d["angr"] = np.nan
            d["l1r"] = np.nan
            d["l2r"] = np.nan
        else:
            vr, tp = rr
            d["angr"] = float(np.arccos(np.clip(np.abs(vs @ vr), -1.0, 1.0)))
            d["l1r"] = float(crossings(np.array([float(vr @ M[0] @ vr)]),
                                       np.array([float(vr @ M[1] @ vr)]))[0])
            d["l2r"] = float(lam2_batch(vr[None, :], M)[0])
        rows.append(d)
    if not rows:
        print(f"   {tag:>26}: no usable draw")
        return
    a = lambda k: np.array([r[k] for r in rows])
    ok_a = int(np.sum(a("rel") <= 1e-10))
    ok_b = int(np.sum(np.abs(a("l1s") - a("true")) / a("gap") <= 1e-8))
    worst_b = float(np.max(np.abs(a("l1s") - a("true")) / a("gap")))
    res_c = (a("l2s") - a("true")) / a("gap")
    print(f"   {tag:>26}: n={len(rows):3d} dimE0={sorted({r['m0'] for r in rows if 'm0' in r})}"
          f"   [A] identity residual <=1e-10 in {ok_a}/{len(rows)}, max={a('rel').max():.1e}"
          f"   [A2] p-block residual max={a('relx').max():.1e}"
          f"   [B] |lambda_1(v*)-lam_min|/gap <=1e-8 in {ok_b}/{len(rows)}, worst={worst_b:.1e}"
          f"   [C] (lambda_2(v*)-lam_min)/gap med={np.median(res_c):.1e} max={np.max(res_c):.1e}"
          f"   side viol={int(np.sum(a('l2s') < a('true') - a('floor')))}"
          f"   B(v*)={a('bstar').mean():.6f} (min over draws {a('bstar').min():.6f})")
    gf1 = np.array([(r["cf1"] - r["l1r"]) / r["gap"] for r in rows if np.isfinite(r["l1r"])])
    gf2 = np.array([(r["cf1"] - r["l2r"]) / r["gap"] for r in rows if np.isfinite(r["l2r"])])
    # [B] FAILED as registered (lam_1(v*) is NOT lam_min), so the quantity that matters is printed
    # explicitly instead: how far the n=1 member sits above the certificate AT THE IDEAL DIRECTION,
    # (lam_1(v*)-lam_min)/gap, and whether v* even beats v1 there, (cf1-lam_1(v*))/gap.  If the first
    # is O(1) and the second O(0), cf1's looseness is TRUNCATIONAL and the whole direction leg (b48/49)
    # was chasing a term the n=1 rule cannot remove at any direction -- which is a different, and much
    # stronger, closing statement than "cf1 is a saddle".
    r1 = (a("l1s") - a("true")) / a("gap")
    r1v = (a("cf1") - a("l1s")) / a("gap")
    r2 = (a("l2s") - a("true")) / a("gap")
    print(f"{'':>28}  [B2] (lambda_1(v*)-lam_min)/gap med={np.median(r1):.4f}"
          f" min={np.min(r1):.4f} max={np.max(r1):.4f}"
          f"   (cf1-lambda_1(v*))/gap med={np.median(r1v):.2e}"
          f" max={np.max(r1v):.2e}   [C2] (lambda_2(v*)-lam_min)/gap med={np.median(r2):.1e}")
    print(f"{'':>28}  [B3] c2 - B(v*): med={np.median(a('c2') - a('bstar')):.2e}"
          f" min={np.min(a('c2') - a('bstar')):.2e} max={np.max(a('c2') - a('bstar')):.2e}"
          f"   |v* . top_evec(A0)| med={np.median(a('top')):.6f} min={np.min(a('top')):.6f}"
          f"   cf1 > lambda_1(v*) in {int(np.sum(a('cf1') > a('l1s')))}/{len(rows)} draws")
    print(f"{'':>28}  [D] angle(v*, v1): med={np.median(a('ang')):.2e}"
          f" p10={np.percentile(a('ang'), 10):.2e} p90={np.percentile(a('ang'), 90):.2e}"
          f"   [E] angle(v*, v_ray): med={np.median(a('angr')):.2e}"
          f"   gapfrac(lambda_1(ray)) med={np.median(gf1):.4f}"
          f"   gapfrac(lambda_2(ray)) med={np.median(gf2):.4f}")


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
print("[A-E] min over unit v of lambda_1(v) = lam_min(G), attained at the normalised q-block of G's")
print("      minimal eigenvector (derived above from the block eigen-equations alone).  Tests A and B")
print("      are the gate; D compares angle(v*,v1) with b52's boundary-layer width t_hi, which is a")
print("      SECOND, INDEPENDENT computation of the same number -- if they disagree the reconciliation")
print("      of b51 with b50 is incomplete and the boundary-layer story is only partly right.")
for name, cs in SPECTRA.items():
    run_cell(f"q=3 {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(6)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "q=3 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "q=3 30-88-89 mix")]:
    run_cell(nm, mixgen(cs, delta))
run_cell("q=2 spread 5-40", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(5)), np.cos(np.deg2rad(40))])))(lam_spd(5)))
run_cell("q=4 spread 5-75+", lambda: (lambda L: (L, blocks(L, [np.cos(np.deg2rad(a)) for a in (5, 25, 55, 80)])))(lam_spd(7)))

print("\n" + "=" * 124)
print("判读（预登记，跑前写）：")
print("  A 恒等式（不含任何求根公式）：||f_lam y - (1-lam)y||/((1-lam)||y||) <= 1e-10 全部成立；")
print("     A2 再用同一个 y 检查 p-块满足 (Lam-lam I)x = -Ky。两条不过则推导错，不是数值问题。")
print("  B lam_1(v*) = lam_min(G)：|lam_1(v*)-lam_min|/gap <= 1e-8 每格全中；某格 >1e-6 说明 eq:(20)")
print("     的另一支在交叉处捣乱（那就是分支错误，不是松紧问题）。")
print("  C 定序在 v*：lam_min-floor <= lam_2(v*) <= lam_1(v*)+floor。由 B 与逐点 lam_2<=lam_1 被迫")
print("     等于 lam_min；若 (lam_2(v*)-lam_min)/gap > 1e-4，则 b50 只在 v1 的 240 个点上测过的")
print("     '逐点定序'有例外 —— 那是有价值的失败，不是噪声。")
print("  D 角度：angle(v*,v1) 中位应落在 [1e-5, 2e-3]，与 b52 的 t_hi 是同量的第二次独立计算；")
print("     >1e-1 则边界层解释不完整。")
print("  E 射线离 v* 有多远：angle(v*,v_ray) 与同一抽样上 gapfrac(lambda_1(ray))。此条只登记为测量，")
print("     不登记预测（记忆规则 8：不能给余项定号的东西不写方向）。")
