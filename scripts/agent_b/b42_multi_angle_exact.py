"""
b42: does Theorem 3.3 (thm:exact) need the single-angle construction, or only the Gram structure?

PAPER DERIVATION (this script only tests it).  Write the operator Gram with a GENERAL coupling
block,  Gr = [[Lam, K],[K^T, I_q]],  Lam spd (p x p), K (p x q) full column rank.  For
lam < hi := min(lam_min(Lam), 1) the Schur complement of Gr - lam*I in the top-left block is
    S(lam) = (1-lam) I_q - K^T (Lam - lam I_p)^{-1} K,
strictly decreasing in the Loewner order on (0,hi): both terms decrease.  So

    lam_min(Gr) = min( hi, lam* ),   lam* = unique root of
        h(lam) := (1-lam) - lam_max( K^T (Lam - lam I)^{-1} K ) = 0            (eq:multi)

with the conventions: no root in (0,hi) -> lam_min = hi;  h(0)=0 -> lam_min = 0 (the two
subspaces intersect).  Cost: a monotone 1-D root find on a q x q symmetric eigenvalue --
still no eigenvalue of the (p+q)-dimensional Gram, exactly as advertised for thm:exact.

RECOVERY of eq:exact.  Put K = cos(a) H, C := Lam^{-1}H, and assume the Ch.4 isometry
C^T Lam C = I_q (equivalently H^T Lam^{-1} H = I_q).  Using Lam(Lam-lam I)^{-1} = I + lam(Lam-lam I)^{-1}:
    K^T(Lam-lam I)^{-1}K = cos^2 a C^T Lam^2 (Lam-lam I)^{-1} C
                        = cos^2 a [ C^T Lam C + lam C^T Lam(Lam-lam I)^{-1} C ]
                        = cos^2 a ( I_q + lam B(lam) ),      B(lam) = C^T Lam (Lam-lam I)^{-1} C
so lam_max = cos^2 a (1 + lam lam_max(B)) and h(lam)=0 becomes
    1 - lam = cos^2 a (1 + lam lam_max(B))  <=>  lam + cos^2 a * lam * lam_max(B(lam)) = sin^2 a.

WHAT IS ACTUALLY AT STAKE.  The isometry says all q principal cosines are EQUAL to cos(a): the
"single angle" of Ch.4 is not a modelling choice about dictionaries, it is an algebraic property
of the constructed examples.  The general object is a SPECTRUM c_1 >= ... >= c_q of cosines
(M = Lam^{-1/2}K has exactly those singular values).  Three predictions, ordered as a referee
would attack them:
  T1 regression: equal-cosine spectrum -> eq:multi, eq:exact and eigvalsh(Gr) must all agree to
     round-off.  If not, my derivation or this code is wrong (and nothing else matters).
  T2 generalisation: spread spectrum -> eq:multi must still hold to round-off.  Special case
     Lam = I gives the closed answer lam_min = 1 - c_1, an independent check of the root find.
  T3 falsification of the scalar story: cf(t,alpha) (prop:plane) is a function of the SCALAR
     t = ||Lam^{-1}K||_2^2 / c_1^2 alone.  With the most generous alpha available (alpha read off
     the TOP cosine), cf must err badly; and cf has a ceiling cf(t,90 deg), above which NO alpha
     fits -- draws past that ceiling are scalar-formula failures for reasons of principle.
"""
import numpy as np

rng = np.random.default_rng(73731)
D = 100


def lam_spd(p):
    """Gram of p unit-norm dictionary vectors: general, well-conditioned, diag = 1."""
    E = rng.normal(size=(D, p))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E.T @ E


def blocks(Lam, cosines):
    """Gr blocks with a PRESCRIBED principal-cosine spectrum (equal list => the Ch.4 construction)."""
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))       # p x q orthonormal columns
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))       # q x q orthogonal
    M = Q1 @ np.diag(cosines) @ Q2.T                    # = Lam^{-1/2} K, singular values = cosines
    ev, V = np.linalg.eigh(Lam)
    Lam_half = V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T
    K = Lam_half @ M
    return K, M


def lam_min_multi(Lam, K, iters=120):
    """eq:multi."""
    hi = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    Ip = np.eye(Lam.shape[0])

    def h(lam):
        return (1.0 - lam) - np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam - lam * Ip, K))[-1]

    lo = 1e-13
    if h(lo) <= 0.0:
        return 0.0
    upper = hi * (1.0 - 1e-12)
    if h(upper) > 0.0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + upper)
        if h(mid) > 0.0:
            lo = mid
        else:
            upper = mid
    return 0.5 * (lo + upper)


def lam_min_exact(Lam, K, iters=120):
    """eq:exact, Ch.4 form.  Returns None unless the isometry (all cosines equal) holds."""
    c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))
    if c2[-1] - c2[0] > 1e-8 * max(1.0, c2[-1]):        # spread spectrum: not statable
        return None
    cos2 = c2[-1]
    hi = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    Ip = np.eye(Lam.shape[0])
    # theorem's C is Lam^{-1}H with K = cos(a) H; solve(Lam,K) carries an extra cos(a) factor.
    C = np.linalg.solve(Lam, K) / np.sqrt(cos2)

    def psi(lam):
        B = C.T @ Lam @ np.linalg.solve(Lam - lam * Ip, C)
        return lam * (1.0 + cos2 * np.linalg.eigvalsh(B)[-1]) - (1.0 - cos2)

    lo, upper = 1e-13, hi * (1.0 - 1e-12)
    if psi(lo) > 0.0:
        return 0.0
    if psi(upper) < 0.0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + upper)
        if psi(mid) < 0.0:
            lo = mid
        else:
            upper = mid
    return 0.5 * (lo + upper)


def cf_scalar(Lam, K, alpha_deg):
    """prop:plane closed form, as a function of the scalar t only (alpha supplied by the caller)."""
    c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))[-1]      # cos^2 of the top cosine
    t = np.linalg.norm(np.linalg.solve(Lam, K), 2) ** 2 / c2        # = ||Lam^{-1}H||^2, H = K/cos
    s = np.sin(np.deg2rad(alpha_deg)) ** 2
    disc = max((1.0 + t) ** 2 - 4.0 * t * s, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t) if t > 0 else s


SPECTRA = {
    "equal 20d":        [np.cos(np.deg2rad(20))] * 3,
    "equal 5d":         [np.cos(np.deg2rad(5))] * 3,
    "spread 5-75":      [np.cos(np.deg2rad(a)) for a in (5, 40, 75)],
    "spread 2-45-88":   [np.cos(np.deg2rad(a)) for a in (2, 45, 88)],
    "spread 30-88-89":  [np.cos(np.deg2rad(a)) for a in (30, 88, 89)],
}

print("=" * 104)
print("[T1/T2] eq:multi vs eigvalsh(Gr), and eq:exact where statable.  p=4, q=3, 80 draws/cell.")
print(f"   {'spectrum':>15}  {'max|multi/true-1|':>18}  {'max|exact/true-1|':>18}  {'statable':>9}  {'mean|true-1|':>12}")
for name, cs in SPECTRA.items():
    em, ee, ok, tr = [], [], 0, []
    for _ in range(80):
        Lam = lam_spd(4)
        K, M = blocks(Lam, cs)
        Gr = np.block([[Lam, K], [K.T, np.eye(len(cs))]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 1e-12:
            continue
        em.append(abs(lam_min_multi(Lam, K) / true - 1))
        pe = lam_min_exact(Lam, K)
        if pe is not None:
            ok += 1
            ee.append(abs(pe / true - 1))
        tr.append(true)
    s_ee = f"{max(ee):18.3e}" if ee else f"{'not statable':>18}"
    print(f"   {name:>15}  {max(em):18.3e}  {s_ee}  {ok:9d}  {np.mean(tr):12.4f}")

print()
print("[T2b] independent check, Lam = I_p (then Gr = [[I,K],[K^T,I]] has lam_min = 1 - c_1 exactly).")
for name, cs in SPECTRA.items():
    errs = []
    for _ in range(40):
        Lam = np.eye(4)
        K, M = blocks(Lam, cs)
        Gr = np.block([[Lam, K], [K.T, np.eye(len(cs))]])
        true = np.linalg.eigvalsh(Gr)[0]
        errs.append(max(abs(lam_min_multi(Lam, K) - true), abs((1 - max(cs)) - true)))
    print(f"   {name:>15}  max|eq:multi-true| = max|1-c1-true| = {max(errs):.3e}")

print()
print("[T3] the SCALAR closed form cf(t,alpha) in the spread regime (p=4, q=3, 120 draws/cell).")
print("     alpha is given the most generous value available: the TOP cosine.  Ceiling = cf(t,90d).")
print("     'viol' = draws with cf < lam_min, i.e. where cf stops being an UPPER estimate of lam_min")
print("     (prop:plane's bound direction is proved only under the isometry, so this is a real test).")
print(f"   {'spectrum':>15}  {'median cf/true-1':>16}  {'min':>9}  {'max':>9}  {'viol':>6}  {'ceiling/true':>12}")
for name, cs in SPECTRA.items():
    sgn, above, ratio = [], 0, []
    for _ in range(120):
        Lam = lam_spd(4)
        K, M = blocks(Lam, cs)
        Gr = np.block([[Lam, K], [K.T, np.eye(len(cs))]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 1e-12:
            continue
        a_top = np.degrees(np.arccos(min(1.0, max(cs))))
        sgn.append(cf_scalar(Lam, K, a_top) / true - 1)
        ceil = cf_scalar(Lam, K, 90.0)
        if true > ceil:
            above += 1
        ratio.append(ceil / true)
    sgn = np.array(sgn)
    print(f"   {name:>15}  {np.median(sgn):16.4f}  {sgn.min():9.4f}  {sgn.max():9.4f}  "
          f"{int((sgn<0).sum()):6d}  {np.median(ratio):12.3f}"
          + (f"   ({above} past the ceiling)" if above else ""))

print()
print("判读：")
print("  T1 equal-cosine rows: both columns at 1e-12 -> eq:multi == eq:exact on the Ch.4 construction")
print("     (regression: the generalisation does not contradict the theorem it contains).")
print("  T2 spread rows: 'statable'=0 -> eq:exact literally cannot be evaluated (its B(lam) is not")
print("     scalar-reducible); if eq:multi is still 1e-12 there, the general form is the true theorem.")
print("  T3 spread rows with median/max cf error O(1) -> prop:plane's t-only closed form is WRONG in the")
print("     multi-angle regime, not merely loose.  Equal-cosine rows must show small error (sanity).")
