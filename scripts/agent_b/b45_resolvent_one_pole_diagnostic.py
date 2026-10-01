"""
b45 -- what cf actually throws away, and a sign law that can kill my own derivation.

Trigger (the 41.8 item, done as DERIVATION first, not as curve-fitting).  b44 established that the
residual of cf is an anisotropy-of-Lam effect and left open "which functional of Lam beyond t
controls it".  Writing eq:multi down and comparing it to eq:cf answers the shape of that question:

  eq:multi:   lambda_min(Gr) = min(lbar, lam*),  h(lam) = (1-lam) - f(lam),  f(lam) := lam_max(K^T (Lam - lam I)^-1 K),
              on (0, lbar), lbar := min(lam_min(Lam), 1).
  eq:cf:      cf = smaller root of q(lam) = (1-lam)(1 - t*lam) - c^2 = 0,  c^2 := lam_max(K^T Lam^-1 K),
              t := ||Lam^-1 K||_2^2 / c^2.

  (1) c^2/(1 - t*lam) is a legitimate name for the cf equation:  (1-lam) = c^2/(1-t*lam)  <=>  q(lam)=0,
      and q(1/t) = -c^2 <= 0 puts cf STRICTLY BELOW the surrogate's pole 1/t.  So on (0, 1/t) the
      function g(lam) := c^2/(1 - t*lam) is increasing and g(cf) = 1 - cf.
      Therefore, with F(lam) := (1-lam) - f(lam) (strictly decreasing on (0,lbar), F(lam*)=0):

          F(cf) = g(cf) - f(cf) = -(f(cf) - g(cf)) =: -D(cf),
          cf > lam*  <=>  F(cf) < 0  <=>  D(cf) > 0                      [SIGN LAW]

      provided lam_min(Gr) = lam* (uncapped) and cf < lbar (so f(cf) exists).
  (2) What is D?  Diagonalise Lam = sum_i lam_i |u_i><u_i| and put k_i := K^T u_i in R^q.  Then
      f(lam) = lam_max( sum_i k_i k_i^T / (lam_i - lam) ), and if v is a unit top eigenvector of
      K^T(Lam-lam I)^-1 K, the scalar function lam |-> sum_i (k_i.v)^2/(lam_i - lam) agrees with f at v
      and hence equals f wherever the top eigenvalue is simple (Hellmann-Feynman).  So

          f(lam) = SUM_i w_i / (lam_i - lam),   w_i := (k_i . v)^2 >= 0,

      i.e. f is the STIELTES TRANSFORM of a positive measure nu = sum_i w_i delta_{lam_i} supported on
      Lam's spectrum.  cf's surrogate replaces nu by the single atom c^2*t at 1/t -- it keeps
      int dnu/x = c^2 and uses int dnu/x^2 <= ||Lam^-1 K||_2^2 = c^2 t (max over v, while f uses the
      TOP v of K^T Lam^-1 K, which need not be the same vector).  So the answer to 41.8 is structural:
      nothing that is a scalar function of Lam alone can control the gap; what controls it is the
      JOINED data (Lam's eigenvalues, the weights with which K probes them), and cf is the
      one-pole collapse of that measure.
  (3) Linearising F at lam* gives the MAGNITUDE LAW  cf - lam* = D(cf)/|F'(lam*)| * (1 + o(1)),
      |F'(lam)| = 1 + SUM_i w_i/(lam_i - lam)^2.  And since f(0) = g(0) = c^2 while
      f'(0) = sum_i w_i/lam_i^2 <= max_u sum_i (K_i u)^2/lam_i^2 = c^2 t (Cauchy-Schwarz / same
      eigenbasis argument), the leading term of D at 0 is -cf*slack with the SLACK
          slack := c^2 t - sum_i w_i/lam_i^2 >= 0,
      so the FIRST-ORDER LAW IS SIGNED:  cf - lam* ~ -cf*slack/|F'(lam*)| <= 0.  Two consequences,
      both testable: (i) cells with resolved POSITIVE gaps must be higher order in cf -- the leading
      term can never produce them; (ii) slack == 0 iff the same index maximises sum(K_i u)^2/lam_i and
      sum(K_i u)^2/lam_i^2.  When Lam's eigenbasis IS K's left singular basis, k_i = sigma_i v_i and
      both maxima are index-wise, so slack == 0 unless the weak atom's 1/lam^2-boost flips the ranking,
      which needs delta >~ 1 - sigma_w/sigma_top.  At b44/T4's and B4's delta (1e-3, 1e-6) that is far
      out of reach, so the real reason those SVD-aligned cells sat at the floor is slack == 0 -- not
      merely "one atom".  Testable: slack must be 0 at machine for the aligned choices (top/mid/low/ker)
      and STRICTLY POSITIVE with a floor-clearing gap for a misaligned choice (mix).  If 'mix' also
      gives slack 0, my reading of (2) is wrong and this Remark cannot be written.

Pre-registered, before looking at any output (seed 47813):
  B0  (does the flip really need a second cosine?  chapter4 line ~1270 asserts it does.)
      q=1 cells: one cosine only, general Lam.  Prediction: NO flip -- the sign of cf - lambda_min is
      constant over draws.  Falsifier: a q=1 cell whose violating draws are RESOLVED (gap above its
      own 100*eps*||Gr|| floor).  Then "needs a second substantial cosine" is wrong as stated and
      rem:multi(ii) + that sentence must be re-checked; the mechanism (2) says why a single atom
      should be one-sided, so I would expect this to fail loudly rather than quietly.
  B1  (the sign law is a THEOREM test, not a fit.)
      Over b44's five spectra x general Lam, plus the isotropic column: count draws where sgn(cf-lam*)
      differs from sgn(D(cf)).  Prediction: exactly 0 mismatches among uncapped, cf<lbar draws.
      Falsifier: one mismatch with |D| above the floor kills (1), so the Remark cannot be written.
  B2  (magnitude.)  ratio (cf-lam*) / (-D(cf)/|F'(lam*)|) -> 1.  Prediction: median within a few
      percent where the gap is small; the flip cells are allowed to be worse (second order in D).
      Falsifier: ratio far from 1 systematically (say med |R-1| > 0.3) => the diagnostic is a sign
      oracle only and I must NOT advertise it as a predictor.
  B3  (isotropy null -- this is cor:iso on trial again.)  At Lam = gamma*I, f == g IDENTICALLY, so
      D(cf) must be 0 at machine precision for every spectrum and every gamma.  Falsifier: |D| >> eps
      => cor:iso's proof has a hole I have not found.
  B4  (the alignment control, done RIGHT this time -- b44/T4 was a null experiment.)
      b44 rotated Lam and let the random Q1 absorb the rotation, so its "aligned" and "transverse"
      cells were one ensemble.  Here K is drawn FIRST and Lam's weak eigenvector u_w is placed
      relative to K's own LEFT singular structure, which is what (2) says is load-bearing:
        aligned    : u_w = top LEFT singular vector of K   -> w at lam_min(Lam) is MAXIMAL,
        transverse : u_w in ker K^T (non-empty since p=4 > q=3) -> w there is EXACTLY 0.
      Lam = V diag(1,1,1,1-delta) V^T, delta = 1e-3, 1e-6.
      Sharper, and stated BEFORE running: in the TRANSVERSE cell the weak eigenvector u is in ker K^T,
      so K^T Lam^{-j} K = sum_{i != weak} k_i k_i^T for every j >= 1, hence
          c^2 = lam_max(K^T K),   t = lam_max(K^T K)/c^2 = 1,   f(lam) = lam_max(K^T K)/(1-lam) = g(lam),
      i.e. D == 0 IDENTICALLY and cf is exact for every cosine spectrum -- K cannot see that direction
      of Lam at all.  So the transverse cell is not merely "smaller", it is a second, independent proof
      test of cor:iso reached from the measure side (any atom of zero weight may be moved freely).
      Prediction: transverse max|gap| and max|D| at the eps floor, t == 1 to round-off, for BOTH delta.
      If instead the transverse gap grows with delta, then something other than the weights enters
      (i.e. kappa(Lam) itself matters), (2) is incomplete, and I must not write the Remark.
      If the two cells are statistically identical AGAIN, then the aligned cell is also inert and the
      control is still broken -- a third null, which I would report as such rather than as evidence.

  --- appended AFTER B0-B4 had been seen, BEFORE running B4b/B5 (so these are still predictions): ---
  B4b the top/ker cells of B4 both came out at the floor, and the reason is now clear from (2): each
      puts ALL of the weight on ONE atom (top: w = (sigma_1^2,0,0,0); ker: the weak atom has w=0), so
      both are exactness cases by construction, not nulls.  The informative cells are the PARTIAL ones
      (weak atom carrying the #2, #3 or a mixed direction's weight) at delta large enough to leave the
      floor.  Prediction: gap > 0 order delta-weighted for #3/#2, and cf < lambda_min possible.
  B5  chapter4 line ~1270 and rem:multi(ii) say the flip "additionally needs a second substantial
      cosine".  b44's five spectra move the top and second cosine together, so that claim is
      CONFOUNDED.  A 2x2 in (top, second) at FIXED geometry separates it from the competing reading
      implied by (1)-(2): the flip is about cf being small (top cosine near 1), which is what lets the
      Cauchy-Schwarz inequality f'(0) <= c^2 t be the leading term of D.  The two readings predict
      OPPOSITE off-diagonal cells, so whichever way it lands one of them is wrong -- mine included.
"""
import numpy as np

rng = np.random.default_rng(47813)
N = 100
EPS = np.finfo(float).eps
P = 4

SPECTRA = {
    "equal 20d":       [np.cos(np.deg2rad(20))] * 3,
    "equal 5d":        [np.cos(np.deg2rad(5))] * 3,
    "spread 5-75":     [np.cos(np.deg2rad(a)
                     ) for a in (5, 40, 75)],
    "spread 2-45-88":  [np.cos(np.deg2rad(a)
                     ) for a in (2, 45, 88)],
    "spread 30-88-89": [np.cos(np.deg2rad(a)
                     ) for a in (30, 88, 89)],
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
    K = (V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T) @ M
    return K


def resolvent(Lam, K, lam):
    """f(lam) = lam_max(K^T (Lam-lam I)^-1 K) and its unit top eigenvector v."""
    A = K.T @ np.linalg.solve(Lam - lam * np.eye(Lam.shape[0]), K)
    w, Vv = np.linalg.eigh(A)
    return w[-1], Vv[:, -1]


def diagnostics(Lam, K):
    """Everything the sign/magnitude laws need, in one draw.  None => draw unusable, reason string."""
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    evG = np.linalg.eigvalsh(Gr)
    true = evG[0]
    c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))[-1]
    if c2 > 1.0 + 1e-12:
        return None, "Gr not PSD (top cosine > 1)"
    c2 = min(c2, 1.0)
    if c2 <= 0 or true <= 1e-12:
        return None, "degenerate"
    t = np.linalg.norm(np.linalg.solve(Lam, K), 2) ** 2 / c2
    s = 1.0 - min(c2, 1.0)                       # sin^2 of the TOP cosine: the generous choice
    disc = max((1.0 + t) ** 2 - 4.0 * t * s, 0.0)
    cf = ((1.0 + t) - np.sqrt(disc)) / (2.0 * t)
    lbar = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    floor = 100.0 * EPS * np.linalg.norm(Gr, 2)
    if cf >= lbar:
        return None, "cf beyond lbar"
    if true >= lbar - 1e-10 * max(1.0, lbar):
        return None, "capped branch"
    f_cf, _ = resolvent(Lam, K, cf)
    D = f_cf - c2 / (1.0 - t * cf)
    # The right-derivative of f at 0 is NOT sum_i (k_i.v0)^2/lam_i^2 for an ARBITRARY top vector v0:
    # if A(0) = K^T Lam^-1 K has a degenerate top eigenspace E0 (equal-angle cells), v may rotate
    # freely inside E0 and the correct quantity is the max over unit v in E0,
    #     f'(0+) = lam_max( E0^T (K^T Lam^-2 K) E0 )  <=  lam_max(K^T Lam^-2 K) = c^2 t,
    # so SLACK_EFF := c^2 t - f'(0+) >= 0 always, is 0 IDENTICALLY when E0 = R^q (fully equal angles),
    # and is the only version of the slack that is well defined.  (The first run of this script used
    # one arbitrary v0 and therefore printed a meaningless slack for the equal-angle cells -- the
    # numbers in b44 are unaffected, this is a defect of the b45 diagnostic, caught by its own data.)
    A0 = K.T @ np.linalg.solve(Lam, K)
    w0, V0 = np.linalg.eigh(A0)
    c2_0 = w0[-1]
    E0 = V0[:, w0 >= c2_0 - 1e-9 * max(abs(c2_0), 1.0)]
    B2 = K.T @ np.linalg.solve(Lam, np.linalg.solve(Lam, K))
    fprime0 = np.linalg.eigvalsh(E0.T @ B2 @ E0)[-1]
    slack = c2 * t - fprime0
    _, v_s = resolvent(Lam, K, true)
    d2 = np.linalg.solve(Lam - true * np.eye(p), K) @ v_s
    slope = 1.0 + d2 @ d2                        # |F'(lam*)| = 1 + sum_i w_i/(lam_i-lam*)^2
    # first-order law: D(cf) = -cf*slack + O(cf^2), so  cf - lam* ~ -cf*slack/|F'| <= 0, i.e. the
    # LEADING term is always on the flip side and can never produce a positive gap.  Positive-gap
    # cells must therefore be higher order in cf, and the flip is a competition of the two.
    return dict(true=true, cf=cf, D=D, slope=slope, floor=floor, t=t, c2=c2, slack=slack,
                m0=E0.shape[1], sec=(np.nan if q < 2 else np.sqrt(max(w0[-2], 0.0) / max(c2_0, 1e-300))),
                lin=-cf * slack / slope, lbar=lbar, gap=cf - true,
                pred=D / slope), "ok"


def sign_report(g, d, fl):
    """Sign-law mismatch count.  REPORTED BOTH WAYS on purpose: a mismatch between two quantities that
    are both at the eigvalsh floor is a coin flip, not evidence, so the falsification-relevant number
    is 'res' = mismatches among draws where BOTH |gap| and |D| clear the floor.  (b45's first run
    printed only the raw count and showed 30-39 'mismatches' in cells whose gaps are 1e-16; that
    number is meaningless and the fixed reporting is here, not in the run log.)"""
    both = (np.abs(g) > fl) & (np.abs(d) > fl)
    return int(np.sum(np.sign(g) * np.sign(d) < 0)), int(np.sum((np.sign(g) * np.sign(d) < 0) & both)), int(both.sum())


def sweep(tag, lam_of, spectra, n=N):
    print(f"\n[{tag}]  {n} draws/cell")
    print(f"   {'cell':>22}  {'n_ok':>4} {'skip':>4}  {'mism raw/res/n':>14}"
          f"  {'med|gap|':>9} {'flips':>5}  {'med|R-1|':>8}   med cf   E0   min slack  med|gap/lin|")
    for name in spectra:
        cs = spectra[name]
        rows, nskip, skipwhy = [], 0, {}
        for _ in range(n):
            Lam = lam_of(P)
            K = blocks(Lam, cs)
            rec, why = diagnostics(Lam, K)
            if rec is None:
                nskip += 1
                skipwhy[why] = skipwhy.get(why, 0) + 1
                continue
            rows.append(rec)
        if not rows:
            print(f"   {name:>22}  all skipped {skipwhy}")
            continue
        g = np.array([r["gap"] for r in rows])
        d = np.array([r["D"] for r in rows])
        fl = np.array([r["floor"] for r in rows])
        pr = np.array([r["pred"] for r in rows])
        sl = np.array([r["slack"] for r in rows])
        ln = np.array([r["lin"] for r in rows])
        cfv = np.array([r["cf"] for r in rows])
        m0 = np.array([r["m0"] for r in rows])
        mraw, mres, mn = sign_report(g, d, fl)
        flips = int(np.sum((g < 0) & (np.abs(g) > fl)))
        ok = np.abs(pr) > 10 * fl
        okl = np.abs(ln) > 10 * fl
        print(f"   {name:>22}  {len(rows):4d} {nskip:4d}  {mraw:4d}/{mres:4d}/{mn:4d}"
              f"  {np.median(np.abs(g)):9.2e} {flips:5d}"
              f"  {np.median(np.abs(g[ok]/pr[ok]-1)) if ok.any() else float('nan'):8.2e}"
              f"  cf={np.median(cfv):7.4f} E0={int(np.median(m0))} slack={sl.min():9.2e}"
              f" |gap/lin|={np.median(np.abs(g[okl]/ln[okl])) if okl.any() else float('nan'):8.2e}")
        if skipwhy:
            print(f"   {'':>22}  skipped as: {skipwhy}")


print("=" * 118)
print("[B0] does the sign flip need a second substantial cosine?  q=1 (single cosine), general Lam.")
print("     prediction: zero resolved flips.  top cosine swept 0.999 down to 0.2.")
for cval in [0.999, 0.99, 0.9, 0.7, 0.2]:
    rows, ncap = [], 0
    for _ in range(N):
        Lam = lam_spd(P)
        K = blocks(Lam, [cval])
        rec, why = diagnostics(Lam, K)
        if rec is None:
            if why == "capped branch":
                ncap += 1
            continue
        rows.append(rec)
    g = np.array([r["gap"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    d = np.array([r["D"] for r in rows])
    print(f"   cos={cval:5.3f}  n={g.size:3d} capped={ncap:3d}  med|gap|={np.median(np.abs(g)):9.2e}"
          f"  min gap={g.min():10.2e}  max gap={g.max():10.2e}  resolved flips={int(np.sum((g<0)&(np.abs(g)>fl))):3d}"
          f"  mism raw/res/n={sign_report(g, d, fl)}")

sweep("B1+B2  general Lam (unit-norm Gram)", lam_spd, SPECTRA)

print("\n" + "=" * 118)
print("[B3] isotropy null: at Lam = gamma*I the two curves are the SAME function, so D(cf) == 0 identically.")
print("     This is cor:iso on trial through a different door (a functional identity, not an eigenvalue identity).")
for gam in [1.0, 1.5, 0.6]:
    for name, cs in SPECTRA.items():
        ds, gs, scl = [], [], []
        for _ in range(N):
            Lam = gam * np.eye(P)
            K = blocks(Lam, cs)
            rec, why = diagnostics(Lam, K)
            if rec is None:
                continue
            ds.append(rec["D"])
            gs.append(abs(rec["gap"]))
            scl.append(rec["c2"] / (1.0 - rec["t"] * rec["cf"]))
        ds = np.abs(np.array(ds))
        print(f"   gamma={gam:3.1f} {name:>16}: n={ds.size:3d}  max|D|={ds.max():9.2e}"
              f"  max|D|/scale={np.max(ds)/max(np.mean(scl),1e-300):9.2e}  max|gap|={np.max(gs):9.2e}")

print("\n" + "=" * 118)
print("[B4] alignment, control FIXED: K drawn first, then Lam's weak eigenvector placed relative to K's")
print("     LEFT singular structure (aligned = top left singular vector; transverse = a vector in ker K^T,")
print("     which is exactly zero-weight in the measure of (2)).  kappa(Lam) is IDENTICAL in the two cells,")
print("     so any difference in the gap is the weight pattern and not the condition number.")


def lam_with_weak(K, delta, weak):
    """Lam = V diag(1,1,1,1-delta) V^T with the WEAK eigenvector pinned to `weak` (a unit vector in R^p).

    Same kappa(Lam) in every cell by construction, so any gap difference is the weight pattern
    w_i = (K^T u_i . v)^2 of the measure in (2), not the condition number.
    """
    U = np.linalg.eigh(np.eye(P) - np.outer(weak, weak))[1][:, 1:4]   # 3 vectors orthogonal to `weak`
    Vv = np.column_stack([U, weak])
    ev = np.ones(P)
    ev[-1] = 1.0 - delta
    return Vv @ np.diag(ev) @ Vv.T


def run_cell(tag, cs, delta, which, rescale=True):
    """Draw K = M with prescribed cosines, put Lam's weak eigenvector at left singular direction
    `which` ('mix' = 45deg of #1 and #2, the only genuinely multi-atom choice), and -- if rescale --
    renormalise K so the REALISED top cosine is cs[0].  Without the rescaling the anisotropic Lam pushes
    K^T Lam^-1 K above 1, i.e. Gr out of the PSD cone, and the cell empties out (the first run's
    'all skipped' lines were that, a construction defect and not a result)."""
    rows = []
    for _ in range(N):
        K0 = blocks(np.eye(P), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = {"top": U[:, 0], "mid": U[:, 1], "low": U[:, 2],
                "mix": (U[:, 0] + U[:, 1]) / np.sqrt(2.0), "ker": U[:, 3]}[which]
        Lam = lam_with_weak(K0, delta, weak)
        if rescale:
            ev = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
            K0 = K0 * min(1.0, cs[0] / np.sqrt(ev[-1]))
        rec, why = diagnostics(Lam, K0)
        if rec is not None:
            rows.append(rec)
    if not rows:
        print(f"   {tag:>34}: all skipped")
        return
    g = np.array([r["gap"] for r in rows])
    d = np.array([r["D"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    tt = np.array([r["t"] for r in rows])
    cfv = np.array([r["cf"] for r in rows])
    sec = np.array([r["sec"] for r in rows])
    sl = np.array([r["slack"] for r in rows])
    ln = np.array([r["lin"] for r in rows])
    m0 = np.array([r["m0"] for r in rows])
    pr = np.array([r["pred"] for r in rows])
    ok = np.abs(pr) > 10 * fl
    okl = np.abs(ln) > 10 * fl
    print(f"   {tag:>34}: n={g.size:3d}  med|gap|={np.median(np.abs(g)):9.2e}"
          f"  max|D|={np.abs(d).max():9.2e}  flips(res/naive)={int(np.sum((g<0)&(np.abs(g)>fl))):3d}/{int(np.sum(g<0)):3d}"
          f"  mism={sign_report(g, d, fl)}  max|t-1|={np.abs(tt-1).max():8.2e}"
          f"  med cf={np.median(cfv):7.4f} 2nd={np.median(sec):6.3f} E0={int(np.median(m0))}"
          f" min slack={sl.min():9.2e}  med|gap/lin|={np.median(np.abs(g[okl]/ln[okl])) if okl.any() else float('nan'):8.2e}"
          f"  med|R-1|={np.median(np.abs(g[ok]/pr[ok]-1)) if ok.any() else float('nan'):8.2e}")


print("\n     weak direction = any of K's LEFT singular vectors, or a kernel direction.")
print("     Derived (and this is what the first run's floor-level numbers mean): in EVERY such cell the")
print("     measure has ONE atom -- for weak = u_j the weight sits on the index that maximises both")
print("     sigma_i^2/lam_i and sigma_i^2/lam_i^2, and those coincide unless delta >~ 1 - sigma_j/sigma_1")
print("     -- so f == g and cf is EXACT for top/mid/low/ker alike.  A null in the sense of (2), not in")
print("     the sense of b44/T4: the control works, it just selects a point of the exactness locus.")
for cs_name in ["spread 2-45-88", "equal 5d"]:
    for delta in [1e-3, 1e-6]:
        run_cell(f"{cs_name} d={delta:.0e} top", SPECTRA[cs_name], delta, "top")
print("     weak direction in ker K^T (zero atom -> also exact, the 41.5(a) generalisation of cor:iso):")
for cs_name in ["spread 2-45-88", "equal 5d"]:
    for delta in [1e-3, 1e-6]:
        run_cell(f"{cs_name} d={delta:.0e} ker", SPECTRA[cs_name], delta, "ker")
print("\n[B4b] the DISCRIMINATING cells: is the gap governed by the atom count of the measure (2)?")
print("     Prediction: mid/low stay on the floor (still one atom, slack 0) while 'mix' -- Lam's weak")
print("     eigenvector at 45deg to K's left basis, genuinely multi-atom -- has slack > 0 and a gap that")
print("     clears the floor.  If 'mix' also sits on the floor, (2) does not control the gap and the")
print("     whole mechanism reading collapses to B1's bare identity.")
for cs_name in ["spread 2-45-88", "spread 30-88-89", "equal 20d"]:
    for delta in [1e-2, 3e-2]:
        for which in ["mid", "low", "mix"]:
            run_cell(f"{cs_name} d={delta:.0e} {which}", SPECTRA[cs_name], delta, which)

print("\n" + "=" * 118)
print("[B5] DECISIVE test against my own published sentence (chapter4 ~line 1270, rem:multi(ii)):")
print("     'the sign flip ADDITIONALLY NEEDS a second substantial cosine'.  b44's five spectra vary the")
print("     top cosine and the second cosine TOGETHER, so that sentence is confounded: top in")
print("     {cos2=0.9994, cos30=0.866} x second in {cos45=0.707, cos84=0.10} is a clean 2x2 with the")
print("     GEOMETRY FIXED (weak atom always along K's #2 left singular vector, delta=3e-2).")
print("     Competing predictions on the two OFF-DIAGONAL cells:")
print("       published 'needs a second cosine'  -> flips only where second = 0.707")
print("       this derivation (second-moment)    -> flips only where top    = 0.9994")
print("     because cf small (top cosine near 1) is what lets the inequality f'(0) <= c^2 t, which is")
print("     Cauchy-Schwarz between int dnu/x^2 and the max that defines t, be the LEADING term of D.")
# third cosine fixed at cos(88) = 0.0349 in all four cells
GRID = [("top 0.9994 / second 0.707", [np.cos(np.deg2rad(a)) for a in (2, 45, 88)]),
        ("top 0.9994 / second 0.10 ", [np.cos(np.deg2rad(a)) for a in (2, 84, 88)]),
        ("top 0.866  / second 0.707", [np.cos(np.deg2rad(a)) for a in (30, 45, 88)]),
        ("top 0.866  / second 0.10 ", [np.cos(np.deg2rad(a)) for a in (30, 84, 88)])]
for name, cs in GRID:
    run_cell(f"B5 {name}", cs, 3e-2, "mix")
print("     (B5's first attempt used which='mid' and every cell came out at the floor; the derivation")
print("     says why -- an eigenbasis-aligned weak direction leaves a ONE-atom measure, so slack == 0")
print("     and cf is exact whatever the cosines.  'mix' is the multi-atom choice.)")

print("\n" + "=" * 118)
print("[B6] the threshold, one parameter at a time, in the multi-atom geometry (mix, delta=3e-2).")
print("     (a) top cosine ladder with the second cosine held near 0.71;  (b) second cosine ladder with")
print("     the top held near 0.9994.  The first-order law says the flip needs BOTH slack > 0 (a second")
print("     atom, hence a second cosine) AND cf small enough that cf*slack beats the O(cf^2) terms, so")
print("     the sign should change along BOTH ladders.  If instead the sign changes along (b) but never")
print("     along (a), then cf's size is irrelevant and b44's 'second cosine' reading was right after")
print("     all; if it changes along (a) but never (b), the second cosine is irrelevant.")
for a in (2, 3, 5, 8, 12, 20, 30):
    run_cell(f"B6a top=cos({a}d) / 2nd=cos(45d)", [np.cos(np.deg2rad(x)) for x in (a, 45, 88)], 3e-2, "mix")
for b in (10, 20, 30, 45, 60, 75, 85):
    run_cell(f"B6b top=cos(2d) / 2nd=cos({b}d)", [np.cos(np.deg2rad(x)) for x in (2, b, 88)], 3e-2, "mix")

print("\n判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  B1 的 sign mismatches 必须全为 0，且 |D|<floor 的格只能出现在 gap 本身也 <floor 的格")
print("     （两边同时是地板才算自洽）。若某格 mism>0 而 |D| 远在 floor 之上 => (1) 的推导错，")
print("     整条 sign law 不能进正文，41.8 的'哪个泛函'问题回到原点。")
print("  B2 的 med|R-1| 若在小 gap 格 <=0.05 => 诊断量不只给符号，给量级；若 >0.3 只准写'符号预言机'。")
print("  B3 若 max|D| 与 gap 同量级且都在 eps 级 => cor:iso 通过第二次独立检验（泛函恒等而非特征值恒等）。")
print("     若某格 |D| >> eps => cor:iso 证明里有我没找到的洞，必须再撤一次。")
print("  B4 若 aligned 与 transverse 这次分开 => 控制量确实是'权重 vs 谱'的联合，不是 kappa(Λ)；")
print("     若再次撞在一起 => alignment 作为非效应关闭（这次是有有效对照的关闭，不是 §41.5(a) 那种假对照）。")
print("     注：B4 的 mism 只算 resolved（两侧都在地板时符号是硬币，不能当反例）。")
print("  B4b 若 partial-weight 三格（mid/low/mix）里 gap 离开地板，则'单原子'解释得到支持：")
print("     cf 精确 ⟺ 测度 ν 单原子 ⟺ Λ 的弱方向要么在 ker K^T、要么在 K 的顶 left 方向。")
print("  B5 两个竞争读数在 off-diagonal 两格给相反预测，必有一错、可能两错：")
print("     若 flip 只出现在 second=0.707 两格 => 我正文那句话是对的（但理由仍要换成测度语言）；")
print("     若 flip 只出现在 top=0.9994 两格 => 正文那句是**误标（第二次）**，必须改成'cf 足够小'")
print("     的条件；若四格都翻或都不翻 => 两个读数都错，且这个'解释'本身要撤，只保留 B1 的恒等式。")
