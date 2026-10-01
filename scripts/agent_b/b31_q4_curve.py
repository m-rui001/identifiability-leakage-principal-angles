# B round 18c (B-CLAIM-41, full alpha curve): q4 = sqrt(LB/lam_min(Gcal)) is a DICTIONARY-ONLY
# quantity -- no data, no n, no sigma. Two things to test separately:
#  (T1) per-sample: does lam_min(Gcal) attain the upper branch?  <=> q4_exact / q4_pred_same -> 1
#  (T2) aggregate:  does the curve q4_pred(alpha) reproduce b28's MEASURED q4 (which came out of a
#      least-squares fit and is reported per (n,alpha) cell)?  b30 checked alpha=2,85; b28 also has 20.
import numpy as np

rng = np.random.default_rng(21)
d, P, Q = 10, 4, 3
orth = lambda A: np.linalg.qr(A)[0]
NS = 400

out = ["=== T1/T2: d=%d p=%d q=%d, %d dictionaries per alpha ===" % (d, P, Q, NS)]
out.append("%6s %9s | %8s %8s %8s | %8s %8s %8s | %8s" %
           ("alpha", "medianC2", "q4_exact", "q4_pred", "ratio", "min r", "max r", "violUB", "b28 meas"))
B28 = {2: (0.9482, 0.9476), 20: (0.9531, 0.9528), 85: (0.9996, 0.9996)}
for alpha_deg in (1, 2, 5, 10, 15, 20, 30, 45, 60, 70, 85):
    alpha = np.deg2rad(alpha_deg)
    ca, sa = np.cos(alpha) ** 2, np.sin(alpha) ** 2
    qs, ps, c2s, ratios, viol = [], [], [], [], 0
    for _ in range(NS):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE = orth(GE)
        a = orth(QE @ orth(rng.normal(size=(P, Q))))
        br = rng.normal(size=(d * d, Q))
        b = orth(br - QE @ (QE.T @ br))
        FB = np.cos(alpha) * a + np.sin(alpha) * b
        H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)] for i in range(P)])
        Lam = GE.T @ GE
        c2 = np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2
        Gop = np.block([[Lam, np.cos(alpha) * H], [np.cos(alpha) * H.T, np.eye(Q)]])
        lam = np.linalg.eigvalsh(Gop)[0]
        ub = sa / (1 + ca * c2)
        lb = min(np.linalg.eigvalsh(Lam)[0] / 2, sa / (1 + 2 * ca * c2))
        if lam > ub * (1 + 1e-12):
            viol += 1
        if lam < lb * (1 - 1e-12):
            viol += 999          # a lower-side violation would be a theorem failure
        q4 = np.sqrt(lb / lam)
        p4 = np.sqrt((1 + ca * c2) / (1 + 2 * ca * c2))
        qs.append(q4); ps.append(p4); c2s.append(c2); ratios.append(q4 / p4)
    m, mp, mr = np.mean(qs), np.mean(ps), np.mean(ratios)
    tag = ""
    if alpha_deg in B28:
        lo, hi = B28[alpha_deg]
        tag = "  [%8.4f..%8.4f]" % (lo, hi)
    out.append("%6d %9.5f | %8.4f %8.4f %8.5f | %8.5f %8.5f %8d | %s" %
               (alpha_deg, np.median(c2s), m, mp, mr, np.min(ratios), np.max(ratios), viol, tag))
print("\n".join(out))
print("")
print("T1 passes if 'ratio' ~ 1 with min/max inside [0.985,1.001] and violUB=0 (upper side attained per sample).")
print("T2 passes if q4_pred (median-C2 curve) sits inside b28's measured n-range in the bracketed columns")
print("     -- b28's q4 came from a least-squares fit, so agreement is NOT by construction.")
