# B round 18c: with lambda_min = UB verified per-sample (b29), the certificate's whole residual
# slack becomes analytic:
#   q4 = sqrt(LB/lambda_min) ~= sqrt(LB/UB) = sqrt((1+cos^2 a |C|^2)/(1+2 cos^2 a |C|^2)),
# i.e. the Young-step ratio, a function of the dictionary only. Compare with b28's measured q4
# (d=10 p=4 q=3: 0.9476-0.9484 at alpha=2 deg, 0.9996 at alpha=85 deg).
import numpy as np

rng = np.random.default_rng(4)


def orth(A):
    return np.linalg.qr(A)[0]


out = []
d, P, Q = 10, 4, 3
c2s = []
for _ in range(400):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    a = orth(QE @ orth(rng.normal(size=(P, Q))))
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(P)] for i in range(P)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)] for i in range(P)])
    c2s.append(np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2)
c2 = np.median(c2s)
out.append("=== d=%d p=%d q=%d: ||Lambda_E^-1 H||_2^2 median=%.5f mean=%.5f q90=%.5f (1/d=%.4f) ==="
           % (d, P, Q, c2, np.mean(c2s), np.quantile(c2s, .9), 1.0 / d))
out.append("%6s %12s %12s %12s" % ("alpha", "q4_pred", "q4_meas(b28)", "diff"))
for a_deg, meas in ((2, 0.9478), (5, None), (20, None), (45, None), (85, 0.9996)):
    ca = np.cos(np.deg2rad(a_deg)) ** 2
    pred = np.sqrt((1 + ca * c2) / (1 + 2 * ca * c2))
    out.append("%6d %12.5f %12s %12s" % (a_deg, pred, meas,
                 "%.5f" % (pred - meas) if meas is not None else "-"))
out.append("")
out.append("if |q4_pred - q4_meas| < 1e-3 the slack of the certificate is fully analytic:")
out.append("it is the eps=1 Young step, and nothing else (no sampling, no noise, no dictionary scale).")
print("\n".join(out))
