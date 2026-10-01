# B round 18b: the chapter's own table prints MEAN lambda_min/sin^2 next to a formula built from
# MEAN ||C||^2, and that comparison "violates" Theorem 4.5's upper bound by ~8%.
# The bound is a per-sample statement, so the mean-vs-mean comparison is not a test.
# Here we test it properly: per-sample violations of BOTH sides, b24's exact setup.
import numpy as np

rng = np.random.default_rng(7)


def orth(A):
    return np.linalg.qr(A)[0]


def build(QE, p, q, alpha, d):
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return a, b, np.cos(alpha) * a + np.sin(alpha) * b


def c2norm(E, a):
    P = len(E)
    d = len(E[0])
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(P)] for i in range(P)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(a.shape[1])] for i in range(P)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2, Lam, H


out = []
d, P, Q = 8, 4, 3
out.append("=== Theorem 4.5 per-sample check, d=%d p=%d q=%d, 300 draws/alpha ===" % (d, P, Q))
out.append("%6s %12s %12s %12s %10s %10s %10s" %
           ("alpha", "mean lam", "mean UB", "viol UB", "mean act/UB", "min act/UB", "viol LB"))
for alpha_deg in (1, 2, 5, 10, 20, 45, 70, 85):
    alpha = np.deg2rad(alpha_deg)
    lams, ubs, cn2s, violU, violL, ratios = [], [], [], 0, 0, []
    for _ in range(300):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        a, b, FB = build(QE, P, Q, alpha, d)
        c2, Lam, H = c2norm(E, a)
        Gop = np.block([[Lam, np.cos(alpha) * H], [np.cos(alpha) * H.T, np.eye(Q)]])
        lam = np.linalg.eigvalsh(Gop)[0]
        ub = np.sin(alpha) ** 2 / (1 + np.cos(alpha) ** 2 * c2)
        lb = min(np.linalg.eigvalsh(Lam)[0] / 2,
                 np.sin(alpha) ** 2 / (1 + 2 * np.cos(alpha) ** 2 * c2))
        lams.append(lam); ubs.append(ub); cn2s.append(c2); ratios.append(lam / ub)
        if lam > ub * (1 + 1e-10):
            violU += 1
        if lam < lb * (1 - 1e-10):
            violL += 1
    out.append("%6d %12.4e %12.4e %6d/300 %12.5f %12.5f %8d/300" %
               (alpha_deg, np.mean(lams), np.mean(ubs), violU, np.mean(ratios), np.min(ratios), violL))
out.append("")
out.append("note: 'mean lam' vs 'mean UB' are NOT comparable (lam and ||C||^2 are sample-correlated);")
out.append("      the per-sample violation count and act/UB are the actual test.")
print("\n".join(out))
