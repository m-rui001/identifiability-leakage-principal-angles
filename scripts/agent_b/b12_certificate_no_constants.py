# B round 10b: payoff test.  Does the CONSTANT-FREE analytic certificate
#   ||Chat - C*||  <=  ( ||Delta||_F + sigma*sqrt(d) ) / sin(alpha)  *  sqrt(1 + 2 cos^2 a ||C||_2^2)  * (1 + kappa sqrt((p+q)/n))
# dominate A's measured errors in his own d10 G1 setting, and what fraction of it is actually used?
# (expectation from Cor 4.4: random Delta dilutes by 1/d, so err/bound ~ 1/d, NOT ~1.)
import numpy as np

rng = np.random.default_rng(11)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha, d):
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return a, b, np.cos(alpha) * a + np.sin(alpha) * b


def Cnorm2(E, a, alpha):
    """||C||_2^2 = ||Lam_E^{-1} H||_2^2,  H_ij = <E_i, a_j>  (data-free, alpha-free)."""
    P, Q = len(E), a.shape[1]
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(P)] for i in range(P)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(len(E[0]), len(E[0])).T) for j in range(Q)] for i in range(P)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2


out = []
d, n = 10, 600
kappa = 0.15

out.append("=== A/d10 G1 replica, p=4 q=3 sigma_noise=1e-4, d=%d n=%d ===" % (d, n))
out.append("%5s %7s %11s %11s %11s %9s %9s %9s"
           % ("alpha", "||D||", "err", "bound", "err/bound", "b*1/d", "||C||^2", "law*d"))
P, Q = 4, 3
for alpha_deg in (2, 5, 10, 20, 30, 45, 60, 90):
    alpha = np.deg2rad(alpha_deg)
    for dn in (0.01, 0.05, 0.10):
        errs, cn2s = [], []
        for _ in range(30):
            E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
            GE = np.column_stack([e.reshape(-1) for e in E])
            QE, _ = np.linalg.qr(GE)
            a, b, FB = build_F_at_angle(QE, P, Q, alpha, d)
            F = [FB[:, j].reshape(d, d) for j in range(Q)]
            cn2 = Cnorm2(E, a, alpha)
            cn2s.append(cn2)
            theta = rng.normal(size=P); psi = rng.normal(size=Q)
            Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * dn
            A = sum(theta[i] * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dh
            U = rng.normal(size=(d, n))
            Y = A @ U + rng.normal(scale=1e-4, size=(d, n))
            G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
            sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
            C_hat = sum(sol[P + j] * F[j] for j in range(Q))
            C = sum(psi[j] * F[j] for j in range(Q))
            errs.append(np.linalg.norm(C_hat - C, "fro"))
        err = np.mean(errs)
        cn2 = np.mean(cn2s)
        sigma = 1e-4
        bound = (dn + sigma * np.sqrt(d)) / np.sin(alpha) \
            * np.sqrt(1 + 2 * np.cos(alpha) ** 2 * cn2) * (1 + kappa * np.sqrt((P + Q) / n))
        out.append("%5d %7.2f %11.4e %11.4e %11.4f %9.4e %9.4f %9.3f"
                   % (alpha_deg, dn, err, bound, err / bound, bound / d, cn2, err * np.sin(alpha) / dn * d))
out.append("")
out.append("=== check: does the FULLY tight worst-case direction reach the bound? (adversarial Delta || u_min) ===")
for alpha_deg in (5, 20, 45):
    alpha = np.deg2rad(alpha_deg)
    tight, rnd = [], []
    for _ in range(20):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        a, b, FB = build_F_at_angle(QE, P, Q, alpha, d)
        F = [FB[:, j].reshape(d, d) for j in range(Q)]
        theta = rng.normal(size=P); psi = rng.normal(size=Q)
        U = rng.normal(size=(d, n))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        Ug, sv, _ = np.linalg.svd(G, full_matrices=False)
        u = Ug[:, -1]
        dn = 0.05
        W = u.reshape(d, n)
        Dm = W @ U.T @ np.linalg.inv(U @ U.T)          # exact solve: Delta U = u_min
        Dm = Dm / np.linalg.norm(Dm, "fro") * dn
        A = sum(theta[i] * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dm
        Y = A @ U + rng.normal(scale=1e-4, size=(d, n))
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        C_hat = sum(sol[P + j] * F[j] for j in range(Q))
        C = sum(psi[j] * F[j] for j in range(Q))
        e = np.linalg.norm(C_hat - C, "fro")
        cn2 = Cnorm2(E, a, alpha)
        bound = (dn + 1e-4 * np.sqrt(d)) / np.sin(alpha) * np.sqrt(1 + 2 * np.cos(alpha) ** 2 * cn2) \
            * (1 + kappa * np.sqrt((P + Q) / n))
        tight.append(e / bound)
        rnd.append(e / dn)
    out.append("alpha=%3d  adv err/bound = %.3f (mean)   [<=1 means the certificate holds]   random-Dir err*sin/d = %.3f"
               % (alpha_deg, np.mean(tight), np.mean(rnd) * np.sin(alpha)))
print("\n".join(out))
