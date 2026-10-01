# B round 10c: does my proposed TWO-SIDED bound on lambda_min(Gop) actually hold?
# lower = min( lam_min(Lam_E)/2 ,  s^2/(1+2 c^2 ||C||_2^2) )      <- the SAFE direction for the certificate
# upper = s^2/(1+ c^2 ||C||_2^2)                                   <- my 17.7 formula (unsafe as a lower bound)
import numpy as np

rng = np.random.default_rng(3)


def orth(A):
    return np.linalg.qr(A)[0]


def setup(d, p, q, alpha):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    F = [np.cos(alpha) * a[:, j].reshape(d, d) + np.sin(alpha) * b[:, j].reshape(d, d) for j in range(q)]
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(p)] for i in range(p)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(q)] for i in range(p)])
    Gop = np.zeros((p + q, p + q))
    Gop[:p, :p] = Lam
    Gop[:p, p:] = np.cos(alpha) * H
    Gop[p:, :p] = np.cos(alpha) * H.T
    Gop[p:, p:] = np.eye(q)
    Cn2 = np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2
    return np.linalg.eigvalsh(Gop)[0], Lam, Cn2, Gop


print("lower <= lam_min(Gop) <= upper   over 25 draws per cell")
print("%4s %6s %12s %12s %12s %9s %9s %10s" % ("d", "alpha", "low/act", "act/up", "lam_min", "lower", "upper", "det-check"))
bad_lo = bad_up = 0
for d in (3, 4, 6, 8, 12, 20):
    for ad in (1, 5, 20, 45, 70, 85):
        a = np.deg2rad(ad)
        s2, c2 = np.sin(a) ** 2, np.cos(a) ** 2
        rows = []
        for _ in range(25):
            lm, Lam, Cn2, Gop = setup(d, 4, 2, a)
            lo = min(np.linalg.eigvalsh(Lam)[0] / 2.0, s2 / (1 + 2 * c2 * Cn2))
            up = s2 / (1 + c2 * Cn2)
            detL = np.linalg.det(Lam) * s2 ** 2
            rows.append((lo / lm, lm / up, lm, lo, up, abs(np.linalg.det(Gop) / detL - 1)))
        v_lo = sum(1 for r in rows if r[0] > 1 + 1e-12)
        v_up = sum(1 for r in rows if r[1] > 1 + 1e-12)
        bad_lo += v_lo
        bad_up += v_up
        print("%4d %6d %12.4f %12.4f %12.4e %12.4e %12.4e %10.1e  viol lo/up=%d/%d"
              % (d, ad, min(r[0] for r in rows), min(r[1] for r in rows),
                 np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]),
                 np.mean([r[4] for r in rows]), np.mean([r[5] for r in rows]), v_lo, v_up))
print("TOTAL violations: lower=%d  upper=%d  out of %d draws" % (bad_lo, bad_up, 6 * 6 * 25))
