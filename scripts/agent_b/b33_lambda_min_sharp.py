# B round 19: a candidate sharper closed form for lambda_min(Gcal).
#   cf(t,alpha) = ((1+t) - sqrt((1+t)^2 - 4 t sin^2 alpha)) / (2t),  t = ||Lambda_E^{-1} H||_2^2
# is the small root of the 2x2 generalized eigenvalue problem restricted to
# span{z*, Lambda_E^{-1} H w*}. This script measures its relative deviation against the exact
# lambda_min(Gcal), compares it with Theorem 4.5's upper branch, and reports the two quantities
# that decide whether the min() in the lower branch is ever active.
# NOT a proof: whether cf is a lower/upper bound for the full (p+q)-dimensional problem is open.
import numpy as np

rng = np.random.default_rng(7)


def orth(A):
    return np.linalg.qr(A)[0]


def build(QE, p, q, alpha, d):
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return a, b, np.cos(alpha) * a + np.sin(alpha) * b


def invariant(E, a):
    P, d = len(E), len(E[0])
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(P)] for i in range(P)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(a.shape[1])]
                  for i in range(P)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2, Lam, H


def cf(t, s2):
    disc = max((1.0 + t) ** 2 - 4.0 * t * s2, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t)


out = []
d, P, Q = 8, 4, 3
DRAW = 300
out.append("=== candidate lambda_min(Gcal) closed form, d=%d p=%d q=%d, %d draws/alpha ===" % (d, P, Q, DRAW))
out.append("sqRt = sqrt(lam/sin^2 alpha) (exact), invm15 = (1+t)^(-1/2) (dictionary-only);")
out.append("the two agree to within the last column, which is the block-mass surrogate used by the")
out.append("chapter's Corollary 'the observed looseness of (star) is one dictionary scalar'.")
out.append("%6s %13s %13s %13s %7s %7s %7s %10s %10s %10s %13s %9s" %
           ("alpha", "max|lam/UB-1|", "max|lam/cf-1|", "mean(lam/cf-1)", "lam>cf", "lam<cf",
            "lam<LB", "mean sqRt", "mean invm15", "max|dev|", "min lam(Lam)/2", "max UB"))
for alpha_deg in (1, 2, 5, 10, 20, 45, 60, 70, 85):
    alpha = np.deg2rad(alpha_deg)
    devU, devC, devs, lamL, ubs = [], [], [], [], []
    sq, iv = [], []
    gt_cf = lt_cf = gt_ub = lt_lb = 0
    for _ in range(DRAW):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        a, b, FB = build(QE, P, Q, alpha, d)
        t, Lam, H = invariant(E, a)
        Gop = np.block([[Lam, np.cos(alpha) * H], [np.cos(alpha) * H.T, np.eye(Q)]])
        lam = np.linalg.eigvalsh(Gop)[0]
        ub = np.sin(alpha) ** 2 / (1 + np.cos(alpha) ** 2 * t)
        lb = min(np.linalg.eigvalsh(Lam)[0] / 2,
                 np.sin(alpha) ** 2 / (1 + 2 * np.cos(alpha) ** 2 * t))
        c = cf(t, np.sin(alpha) ** 2)
        devU.append(abs(lam / ub - 1))
        devC.append(abs(lam / c - 1))
        devs.append(lam / c - 1)
        lamL.append(np.linalg.eigvalsh(Lam)[0] / 2)
        ubs.append(ub)
        gt_cf += lam > c
        lt_cf += lam < c
        gt_ub += lam > ub
        lt_lb += lam < lb
        sq.append(np.sqrt(lam / np.sin(alpha) ** 2))
        iv.append((1.0 + t) ** -0.5)
    devI = np.abs(np.array(sq) - np.array(iv)) / np.array(iv)
    out.append("%6d %13.4e %13.4e %13.4e %7d %7d %7d %10.5f %10.5f %10.4f %13.4f %9.4f" %
               (alpha_deg, max(devU), max(devC), np.mean(devs), gt_cf, lt_cf, lt_lb,
                np.mean(sq), np.mean(iv), devI.max(), min(lamL), max(ubs)))
out.append("")
out.append("lam<cf on every draw is the interesting column: cf is then a candidate UPPER bound, strictly")
out.append("sharper than Theorem 4.5's, and its direction agrees (it never underestimates lambda_min).")
out.append("the min() in the lower branch is never active on this grid if min lam(Lam)/2 > max UB for every alpha.")
out.append("cf is a CANDIDATE: matching a quantity in Monte-Carlo is not evidence that it bounds it.")
print("\n".join(out))
