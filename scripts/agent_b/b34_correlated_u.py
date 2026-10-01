# B round 19b: Limitations item 3 says Gaussian i.i.d. U is LOAD-BEARING for the cancellation q1*q2 ~ 1.
# Here U is a trajectory design: columns are consecutive states of d independent AR(1) series with unit
# stationary variance, x_{t+1} = rho x_t + sqrt(1-rho^2) eps_t. rho=0 is exactly b28's arm.
#
# Prediction written BEFORE running (so the result cannot be re-labelled afterwards):
#   (P1) q1 = sqrt(lam_min(Gcal)/lam_min(G^T G/n)) INCREASES with rho (the design Gram's lower edge is
#        short by (p+q)/n_eff, n_eff ~ n(1-rho)/(1+rho), so it falls faster than the operator bound does);
#   (P2) q2 = ||Delta U||_F/(sqrt(n)||Delta||_F) DECREASES with rho (sigma_min(U) shrinks, so realizing a
#        fixed Delta U costs more Frobenius norm);
#   (P3) the product q1*q2 LEAVES 1 DOWNWARD, i.e. the certificate gets looser, never tighter;
#   (P4) err/bound stays < 1 in every cell (safety direction preserved).
# If P3 fails upward, B-CLAIM-42 (revised) is dead and the certificate needs a calibration term after all.
import numpy as np

rng = np.random.default_rng(21)
d, P, Q = 10, 4, 3
sigma, dn = 1e-4, 0.05
orth = lambda A: np.linalg.qr(A)[0]


def frame_invariant(E, a):
    V = np.column_stack([e.reshape(-1) for e in E])
    Lam = V.T @ V
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)] for i in range(P)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2, Lam, H


def ar1_columns(n, rho):
    X = np.empty((d, n))
    X[:, 0] = rng.normal(size=d)
    s = np.sqrt(max(1.0 - rho * rho, 0.0))
    for t in range(1, n):
        X[:, t] = rho * X[:, t - 1] + s * rng.normal(size=d)
    return X


def trial(alpha, n, rho):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE = orth(GE)
    a = orth(QE @ orth(rng.normal(size=(P, Q))))
    br = rng.normal(size=(d * d, Q))
    b = orth(br - QE @ (QE.T @ br))
    FB = np.cos(alpha) * a + np.sin(alpha) * b
    F = [FB[:, j].reshape(d, d) for j in range(Q)]
    c2, Lam, H = frame_invariant(E, a)
    lam_G = np.linalg.eigvalsh(np.column_stack([GE, FB]).T @ np.column_stack([GE, FB]))[0]
    psi = rng.normal(size=Q)
    U = ar1_columns(n, rho)
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
    lam_e = np.linalg.eigvalsh(G.T @ G / n)[0]
    Ug, sv, Vt = np.linalg.svd(G, full_matrices=False)
    W = Ug[:, -1].reshape(d, n)
    Dm = W @ U.T @ np.linalg.inv(U @ U.T)
    cost = np.linalg.norm(Dm, "fro")
    Dm = Dm / cost * dn
    A = sum(rng.normal() * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dm
    Y = A @ U + rng.normal(scale=sigma, size=(d, n))
    sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
    C_hat = sum(sol[P + j] * F[j] for j in range(Q))
    C = sum(psi[j] * F[j] for j in range(Q))
    e = np.linalg.norm(C_hat - C, "fro")
    LB = np.sin(alpha) ** 2 / (1 + 2 * np.cos(alpha) ** 2 * c2)
    eb = e / (dn / np.sqrt(LB))
    q4 = np.sqrt(LB / lam_G)
    q1 = np.sqrt(lam_G / lam_e)
    q2 = 1.0 / (cost * np.sqrt(n))
    v = Vt[-1][P:]
    q3 = np.linalg.norm(FB @ v, 2)
    return dict(eb=eb, q1=q1, q2=q2, q3=q3, q4=q4, prod=q1 * q2, four=q4 * q1 * q2 * q3,
                smin=sv[-1] / np.sqrt(n), lam_e=lam_e)


out = ["q1*q2 under a correlated (AR(1) trajectory) design, d=10 p=4 q=3, alpha=2 deg, 60 trials/cell"]
out.append("%5s %6s %9s %9s %9s %9s %9s %9s %9s" %
           ("rho", "n", "q1", "q2", "q1*q2", "q4", "q3", "q4q1q2q3", "err/bnd"))
for alpha_deg in (2, 20):
    al = np.deg2rad(alpha_deg)
    for rho in (0.0, 0.5, 0.8, 0.95):
        for n in (200, 600):
            rs = [trial(al, n, rho) for _ in range(60)]
            m = lambda k: np.mean([r[k] for r in rs])
            out.append("%5.2f %6d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" %
                       (rho, n, m("q1"), m("q2"), m("prod"), m("q4"), m("q3"), m("four"), m("eb")))
    out.append("")
print("\n".join(out))
print("alpha=%s deg above. n_eff = n(1-rho)/(1+rho); if q1*q2 stays ~1 the cancellation is NOT Gaussian-specific.")
print("lam_e (not printed) drops with rho; smin(G)/sqrt(n) is the quantity being compared against lam_G.")
