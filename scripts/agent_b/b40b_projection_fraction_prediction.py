# B round 20 (addendum to b40): is A's fitted constant EXACTLY the random-projection fraction?
# If err ~ ||P_F Delta||/sin(alpha) then c1 := err*sin(alpha)/||Delta|| should equal E||P_F Delta||/||Delta||,
# which has a closed form: ||P_F Delta||^2/||Delta||^2 ~ Beta(q/2,(d^2-q)/2), so
#   c1_pred = E[sqrt(X)] = B(q/2+1/2, (d^2-q)/2) / B(q/2, (d^2-q)/2),  X ~ Beta(q/2,(d^2-q)/2),
# and sqrt(q)/d = sqrt(E X) is the naive version. No regression, no dictionary, no alpha in this quantity,
# so it can be computed to arbitrary precision and compared against b40's measured c1.
# Falsifiable: measured/pred must be ~1 (within MC error) if the "physical constant" story is wrong and the
# dilution story is right; a systematic excess would mean the least-squares coupling adds its own factor.
import numpy as np
from scipy.special import betaln

q = 3
meas = {6: 2.4777e-01, 8: 1.9986e-01, 10: 1.5477e-01, 14: 1.1712e-01, 20: 8.6160e-02}
print("%4s %11s %11s %10s %10s %10s" %
      ("d", "sqrt(q)/d", "E[sqrtX]", "meas", "meas/naive", "meas/exact"))
for d in (6, 8, 10, 14, 20):
    a, b = q / 2.0, (d * d - q) / 2.0
    e_sqrt = np.exp(betaln(a + 0.5, b) - betaln(a, b))
    naive = np.sqrt(q) / d
    print("%4d %11.6f %11.6f %10.6f %10.4f %10.4f" %
          (d, naive, e_sqrt, meas[d], meas[d] / naive, meas[d] / e_sqrt))

print()
print("independent MC of E||P_F Delta||/||Delta|| (10000 unit-norm operators, random q-dim subspace):")
rng = np.random.default_rng(2026)
for d in (6, 10, 20):
    a, b = q / 2.0, (d * d - q) / 2.0
    e_sqrt = np.exp(betaln(a + 0.5, b) - betaln(a, b))
    vals = []
    for _ in range(100):
        Dl = rng.normal(size=(d * d, 100)); Dl /= np.linalg.norm(Dl, axis=0)
        F = np.linalg.qr(rng.normal(size=(d * d, q)))[0]
        vals.append(np.linalg.norm(F.T @ Dl, axis=0))
    vals = np.concatenate(vals)
    print("  d=%2d  MC=%.6f (se %.6f)  exact=%.6f  MC/exact=%.4f" %
          (d, vals.mean(), vals.std() / np.sqrt(vals.size), e_sqrt, vals.mean() / e_sqrt))
