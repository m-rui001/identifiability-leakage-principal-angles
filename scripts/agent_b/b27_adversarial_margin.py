# B round 14c: b26 showed the margin form is UNRESOLVABLE with random Delta -- err/bound sits at
# 0.13-0.17 for every n from 60 to 1800 with or without a margin, because the 1/d dilution gives the
# bound ~7x slack and sigma=1e-4 makes the noise term negligible.  The margin only bites in the
# ADVERSARIAL regime where b12 measured err/bound = 0.88-0.92.  Run that arm across n.
# Expectation: raw (no margin) must exceed 1 at small n; new margin should keep it <=1.
import numpy as np

rng = np.random.default_rng(11)
d, P, Q = 10, 4, 3
sigma = 1e-4
dn = 0.05


def orth(A):
    return np.linalg.qr(A)[0]


def Cnorm2(E, a):
    p, q = len(E), a.shape[1]
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(p)] for i in range(p)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(q)] for i in range(p)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2


def margin_old(n):
    return 1 + 0.15 * np.sqrt((P + Q) / n)


def margin_new(n, alpha):
    c = 9.0 if np.rad2deg(alpha) <= 30 else 20.0
    return 1.0 / max(1 - c / (2 * n) - 0.88 / np.sqrt(n), 1e-6)


out = ["=== adversarial Delta: vec(Delta U) parallel to u_min(G), ||Delta||_F = %.2f ===" % dn]
out.append("%6s %6s | %10s %10s %10s | %10s %10s %10s | %6s"
           % ("n", "alpha", "raw", "old", "new", "max raw", "max old", "max new", "viol"))
for n in (30, 60, 100, 200, 600, 1800):
    for alpha_deg in (2, 5, 20, 45, 85):
        alpha = np.deg2rad(alpha_deg)
        rr, oo, nn_ = [], [], []
        for _ in range(120):
            E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
            GE = np.column_stack([e.reshape(-1) for e in E])
            QE = np.linalg.qr(GE)[0]
            a = orth(QE @ orth(rng.normal(size=(P, Q))))
            b_raw = rng.normal(size=(d * d, Q))
            b = orth(b_raw - QE @ (QE.T @ b_raw))
            FB = np.cos(alpha) * a + np.sin(alpha) * b
            F = [FB[:, j].reshape(d, d) for j in range(Q)]
            theta = rng.normal(size=P); psi = rng.normal(size=Q)
            U = rng.normal(size=(d, n))
            G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
            Ug, sv, _ = np.linalg.svd(G, full_matrices=False)
            W = Ug[:, -1].reshape(d, n)
            Dm = W @ U.T @ np.linalg.inv(U @ U.T)
            Dm = Dm / np.linalg.norm(Dm, "fro") * dn
            A = sum(theta[i] * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dm
            Y = A @ U + rng.normal(scale=sigma, size=(d, n))
            sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
            C_hat = sum(sol[P + j] * F[j] for j in range(Q))
            C = sum(psi[j] * F[j] for j in range(Q))
            e = np.linalg.norm(C_hat - C, "fro")
            core = (dn + sigma * np.sqrt(d)) / np.sin(alpha) * np.sqrt(1 + 2 * np.cos(alpha) ** 2 * Cnorm2(E, a))
            rr.append(e / core); oo.append(e / (core * margin_old(n))); nn_.append(e / (core * margin_new(n, alpha)))
        rr = np.array(rr)
        out.append("%-6d %-6d | %10.3f %10.3f %10.3f | %10.3f %10.3f %10.3f | %6.3f"
                   % (n, alpha_deg, rr.mean(), np.mean(oo), np.mean(nn_),
                      rr.max(), np.max(oo), np.max(nn_), (rr > 1).mean()))
out.append("")
out.append("### 'viol' = fraction of trials where the NO-MARGIN certificate is exceeded; the margins must fix it.")
print("\n".join(out))
