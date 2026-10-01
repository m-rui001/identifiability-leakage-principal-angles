# B round 14b: §26.4 payoff test.  Compare three forms of the same constant-free certificate on
# A's d10 G1 setting:
#   (i)  no sample margin                      x1
#   (ii) my OLD margin    1 + kappa*sqrt((p+q)/n),  kappa=0.15   (shape retracted in 23.4)
#   (iii) the NEW margin  1 / (1 - c/2n - 0.88/sqrt(n)),  c=9 for alpha<=30 deg, c=20 otherwise  (26.4)
# and sweep n down to 60, where the two margins differ by 2x.  Report err/bound: <=1 means the
# certificate holds.  Also report the fraction of trials exceeding 1 (the certificate must be valid
# at the 0.5% level, so we need this to be ~0 for 30-120 trials).
import numpy as np

rng = np.random.default_rng(11)
d, P, Q = 10, 4, 3
sigma = 1e-4


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
    r = 1 - c / (2 * n) - 0.88 / np.sqrt(n)
    return 1.0 / max(r, 1e-6)


out = ["=== core term (||D||+sigma*sqrt(d))/sin(a) * sqrt(1+2cos^2(a)||C||^2), margins x1 / old / new ==="]
out.append("%6s %5s %10s %11s %11s %11s %11s %11s %11s"
           % ("n", "alpha", "||D||", "err", "raw", "raw e/b", "old e/b", "new e/b", "max new"))
for n in (60, 100, 200, 600, 1800):
    for alpha_deg in (2, 5, 20, 45, 85):
        alpha = np.deg2rad(alpha_deg)
        for dn in (0.05,):
            errs, cn2s = [], []
            for _ in range(120):
                E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
                GE = np.column_stack([e.reshape(-1) for e in E])
                QE = np.linalg.qr(GE)[0]
                a = orth(QE @ orth(rng.normal(size=(P, Q))))
                b_raw = rng.normal(size=(d * d, Q))
                b = orth(b_raw - QE @ (QE.T @ b_raw))
                FB = np.cos(alpha) * a + np.sin(alpha) * b
                F = [FB[:, j].reshape(d, d) for j in range(Q)]
                cn2s.append(Cnorm2(E, a))
                theta = rng.normal(size=P); psi = rng.normal(size=Q)
                Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * dn
                A = sum(theta[i] * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dh
                U = rng.normal(size=(d, n))
                Y = A @ U + rng.normal(scale=sigma, size=(d, n))
                G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
                sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
                C_hat = sum(sol[P + j] * F[j] for j in range(Q))
                C = sum(psi[j] * F[j] for j in range(Q))
                errs.append((np.linalg.norm(C_hat - C, "fro"),
                             (dn + sigma * np.sqrt(d)) / np.sin(alpha) * np.sqrt(1 + 2 * np.cos(alpha) ** 2 * cn2s[-1])))
            core = np.mean([c for _, c in errs])
            err = np.mean([e for e, _ in errs])
            raw = err / core
            ratios_new = [e / (core * margin_new(n, alpha)) for e, _ in errs]
            out.append("%-6d %-5d %10.2f %11.4e %11.4e %11.4f %11.4f %11.4f %11.4f"
                       % (n, alpha_deg, dn, err, core, raw, raw / margin_old(n),
                          err / (core * margin_new(n, alpha)), max(ratios_new)))
out.append("")
out.append("### margin_old = 1+0.15*sqrt(7/n);  margin_new = 1/(1-c/2n-0.88/sqrt(n));  raw = no margin")
out.append("### 'max new' is the worst of 120 trials (target: the certificate is claimed at the 0.5% level)")
print("\n".join(out))
