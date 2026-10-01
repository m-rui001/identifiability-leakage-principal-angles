# B round 13h: b24 showed E[lambda_min(G^TG/n)]/lambda_min(Gop) - 1 is INDEPENDENT of alpha
# (0.25 deg to 20 deg all give -0.035 at n=200, -0.012 at n=600, -0.003 at n=2400) and that
# rel_bias * n ~= 7.1 = p+q for that cell.  Conjecture (closes G-gamma):
#       E[lambda_min(G^TG/n)] = lambda_min(Gop) * (1 - (p+q)/n + O(1/d) + O((p+q)^2/n^2))
# Note it is (p+q)/n, NOT the Marchenko-Pastur edge (1-sqrt((p+q)/(dn)))^2 = 0.872 here, i.e. the
# effective sample size is n (columns of U), not the ambient dn.
# Falsification: sweep p+q and n; rel_bias*n must track p+q.  Then sweep d for the O(1/d) term.
import numpy as np

RNG = np.random.default_rng(99)


def cell(d, p, q, n, alpha, trials=400):
    rel = []
    for _ in range(trials):
        E = [RNG.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE = np.linalg.qr(GE)[0]
        if q:
            a = np.linalg.qr(QE @ np.linalg.qr(RNG.normal(size=(p, q)))[0])[0]
            br = RNG.normal(size=(d * d, q))
            b = np.linalg.qr(br - QE @ (QE.T @ br))[0]
            GF = np.cos(alpha) * a + np.sin(alpha) * b
            Ms = E + [GF[:, j].reshape(d, d) for j in range(q)]
            big = np.column_stack([GE, GF])
        else:
            Ms = E
            big = GE
        lt = np.linalg.eigvalsh(big.T @ big)[0]
        U = RNG.normal(size=(d, n))
        le = np.linalg.eigvalsh(np.column_stack([(Mi @ U).reshape(-1) for Mi in Ms]).T @
                                np.column_stack([(Mi @ U).reshape(-1) for Mi in Ms]) / n)[0]
        rel.append(le / lt)
    rel = np.array(rel)
    return rel.mean() - 1.0, np.quantile(rel, .005) - 1.0, rel.std()


alpha = np.deg2rad(5.0)
print("=== A. p+q sweep at d=8, n=200/600/1200, alpha=5deg, 400 trials ===")
print("%4s %4s %6s | %10s %10s %12s | %10s" % ("p", "q", "p+q", "n", "rel_bias", "rel_bias*n", "0.5% tail"))
for n in (200, 600, 1200):
    for p, q in ((1, 1), (2, 2), (3, 2), (4, 3), (5, 5), (7, 5), (9, 5)):
        rb, tl, sd = cell(8, p, q, n, alpha)
        print("%-4d %-4d %-6d %6d | %10.5f %10.3f %12.5f | (sd=%.4f)"
              % (p, q, p + q, n, rb, rb * n, tl, sd))

print("")
print("=== B. d sweep (does the law carry an O(1/d)?), p=4 q=3 n=600 alpha=5deg ===")
print("%4s %10s %10s" % ("d", "rel_bias", "rel_bias*n"))
for d in (3, 4, 6, 8, 12, 16, 24):
    rb, tl, sd = cell(d, 4, 3, 600, alpha)
    print("%-4d %10.5f %10.3f" % (d, rb, rb * 600))

print("")
print("=== C. alpha-uniformity of the SAME law at p+q=7, d=8, n=600 ===")
print("%9s %10s %10s" % ("alpha_deg", "rel_bias", "rel_bias*n"))
for ad in (0.1, 0.5, 2, 10, 30, 60, 89):
    rb, tl, sd = cell(8, 4, 3, 600, np.deg2rad(ad))
    print("%-9s %10.5f %10.3f" % (ad, rb, rb * 600))
