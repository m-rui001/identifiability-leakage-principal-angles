# B round 17: B-CLAIM-37, measuring the exact identity behind b27's n-flatness.
# For the adversarial construction (Delta = min-norm preimage of W = reshape(u_min(G)), rescaled to
# ||Delta||_F = dn) the leakage is exact:
#     err = (dn / cost) / sigma_min(G),   cost = ||min-norm Delta with Delta U = W||_F
#     bound = dn / sqrt(LB),   LB = sin^2(a)/(1 + 2cos^2(a)||C||_2^2)   (theorem 4.1 lower side)
# =>  err/bound = sqrt(LB) / (cost * sqrt(n * lam_emp))  =  q4 * q1 * q2,  where
#     q4 = sqrt(LB/lam_G)                 theorem-4.1 slack,        n-free by construction
#     q1 = sqrt(lam_G/lam_emp)            sampling shortfall of the DESIGN Gram (b24/b25: ~1.13 at n=30)
#     q2 = 1/(cost*sqrt(n)) = ||Delta U||/(sqrt(n)||Delta||)   realization factor of the map Delta -> Delta U
# Cancellation is confirmed if q1 rises at small n and q2 falls by the same amount.
import numpy as np

rng = np.random.default_rng(11)
d, P, Q = 10, 4, 3
sigma, dn = 1e-4, 0.05
orth = lambda A: np.linalg.qr(A)[0]


def Cnorm2(E, a):
    p, q = len(E), a.shape[1]
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(p)] for i in range(p)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(q)] for i in range(p)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2


out = ["%6s %6s | %8s %8s %8s %8s | %8s %8s"
       % ("n", "alpha", "err/bnd", "q4", "q1", "q2", "product", "|resid|")]
for n in (30, 60, 100, 200, 600, 1800):
    for alpha_deg in (2, 20, 85):
        alpha = np.deg2rad(alpha_deg)
        acc = {k: [] for k in ("eb", "q4", "q1", "q2", "pr")}
        for _ in range(120):
            E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
            GE = np.column_stack([e.reshape(-1) for e in E])
            QE = np.linalg.qr(GE)[0]
            a = orth(QE @ orth(rng.normal(size=(P, Q))))
            br = rng.normal(size=(d * d, Q))
            b = orth(br - QE @ (QE.T @ br))
            FB = np.cos(alpha) * a + np.sin(alpha) * b
            F = [FB[:, j].reshape(d, d) for j in range(Q)]
            big = np.column_stack([GE, FB])
            lam_G = np.linalg.eigvalsh(big.T @ big)[0]
            theta = rng.normal(size=P); psi = rng.normal(size=Q)
            U = rng.normal(size=(d, n))
            Gc = [(Mi @ U).reshape(-1) for Mi in E + F]
            G = np.column_stack(Gc)
            lam_e = np.linalg.eigvalsh(G.T @ G / n)[0]
            Ug, sv, _ = np.linalg.svd(G, full_matrices=False)
            W = Ug[:, -1].reshape(d, n)                     # ||W||_F = 1
            Dm = W @ U.T @ np.linalg.inv(U @ U.T)
            cost = np.linalg.norm(Dm, "fro")
            q2 = 1.0 / (cost * np.sqrt(n))
            Dm = Dm / cost * dn
            A = sum(theta[i] * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dm
            Y = A @ U + rng.normal(scale=sigma, size=(d, n))
            sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
            C_hat = sum(sol[P + j] * F[j] for j in range(Q))
            C = sum(psi[j] * F[j] for j in range(Q))
            e = np.linalg.norm(C_hat - C, "fro")
            LB = np.sin(alpha) ** 2 / (1 + 2 * np.cos(alpha) ** 2 * Cnorm2(E, a))
            q4 = np.sqrt(LB / lam_G)
            q1 = np.sqrt(lam_G / lam_e)
            eb = e / (dn / np.sqrt(LB))
            acc["eb"].append(eb); acc["q4"].append(q4); acc["q1"].append(q1)
            acc["q2"].append(q2); acc["pr"].append(q4 * q1 * q2)
        m = lambda k: np.mean(acc[k])
        out.append("%-6d %-6d | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f"
                   % (n, alpha_deg, m("eb"), m("q4"), m("q1"), m("q2"), m("pr"),
                      abs(m("eb") - m("pr"))))
print("\n".join(out))
print("")
print("### q4 must be n-free (it is a property of the operator Gram only).")
print("### if q1 rises toward small n while q2 falls by the same factor, 27.3's cancellation is CONFIRMED.")
print("### if q1 is flat, the b24/b25 lambda_min bias does not apply at this conditioning and 27.3 is WRONG.")
