# B round 20 (b41): which story explains the fitted model-form constant c1?
# Chapter Remark 5.x currently says  c1 ~ (1/d) x sqrt(1+2cos^2 alpha*t)  -- but 0.10*1.08 = 0.108, while
# the measured c1 at d=10 is 0.155 (b40: 0.1548). The missing factor is ~1.5, and b40 says it is sqrt(q):
#   (S1) dilution-per-column:      c1 = (1/d)*sqrt(1+2cos^2 alpha*t), INDEPENDENT of q
#   (S2) block projection:         c1 = E||P_F Delta||/||Delta|| = Beta mean, leading term sqrt(q)/d
# A q-scan at fixed d separates them: S1 predicts c1 flat in q, S2 predicts c1/sqrt(q) constant.
# Registered before running. Also record the alpha-dependence: S1 carries a sqrt(1+2cos^2 alpha t) tilt
# (11% between 5 deg and 90 deg at t=0.12), S2 is alpha-blind by construction, so alpha cannot separate
# them at this noise level -- reported as such rather than used as evidence.
import numpy as np
from scipy.special import betaln

rng = np.random.default_rng(71717)


def orth(A):
    return np.linalg.qr(A)[0]


d, p = 10, 6   # p fixed at 6 so that q may range 1..6; neither candidate formula involves p


def build_F_at_angle(QE, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    br = rng.normal(size=(d * d, q))
    b = orth(br - QE @ (QE.T @ br))
    return np.cos(alpha) * a + np.sin(alpha) * b


def run(q, alpha_deg, delta_norm, noise, n_samples=600, trials=100):
    alpha = np.deg2rad(alpha_deg)
    errs = []
    for _ in range(trials):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        FB = build_F_at_angle(QE, q, alpha)
        F = [FB[:, j].reshape(d, d) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * delta_norm
        A = (sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh)
        U = rng.normal(size=(d, n_samples))
        Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        C_hat = sum(sol[p + j] * F[j] for j in range(q))
        C = sum(psi[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - C, "fro"))
    return np.mean(errs)


out = []
out.append("=== S1 vs S2: q-scan at d=10, p=6, alpha=20 deg, ||D||=0.05, sigma=1e-4, n=600 ===")
out.append("%3s %12s %12s %12s %12s %12s" % ("q", "c1", "c1/sqrt q", "Beta mean/d", "c1*d/sqrtq", "c1/exact"))
for q in (1, 2, 3, 4, 6):
    c1 = run(q, 20.0, 0.05, 1e-4) * np.sin(np.deg2rad(20.0)) / 0.05
    a, b = q / 2.0, (d * d - q) / 2.0
    exact = np.exp(betaln(a + .5, b) - betaln(a, b))  # E[sqrt(X)], X ~ Beta(q/2,(d^2-q)/2)
    out.append("%3d %12.4e %12.4f %12.4e %12.3f %12.3f" %
               (q, c1, c1 / np.sqrt(q), exact, c1 * d / np.sqrt(q), c1 / exact))
out.append("S1 (q-blind) would show c1 flat -> c1/sqrt q rising like sqrt q. S2 shows c1*d/sqrtq ~ 1/d-ish")
out.append("and c1/exact ~ 1. 'exact' is E||P_F Delta||/||Delta||, the Beta mean (no fitted parameter).")
out.append("")
out.append("=== the alpha tilt is NOT a discriminator (stated, not used) ===")
out.append("%5s %12s %12s" % ("alpha", "c1", "sqrt(1+2cos^2 a t), t=0.12"))
for a_deg in (5, 20, 45, 80):
    c1 = run(3, a_deg, 0.05, 1e-4, trials=60) * np.sin(np.deg2rad(a_deg)) / 0.05
    tilt = np.sqrt(1 + 2 * np.cos(np.deg2rad(a_deg)) ** 2 * 0.12)
    out.append("%5d %12.4e %12.3f" % (a_deg, c1, tilt))
print("\n".join(out))
