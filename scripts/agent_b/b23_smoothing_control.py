# B round 13e: DECISIVE CONTROLS.  b22 showed the surviving signal is NG(4)~2.1-2.6 and only in
# windows that include ONI (a 3-month running mean of Nino3.4) or MEI (an EOF index).  SST-only and
# SST+SOI give nothing.  ONI/MEI are DETERMINISTIC LINEAR TRANSFORMS of the underlying series, so a
# lag-1 fit on [x, mean(x)] is a fit on a state that is not the Markov state.
# Control A/B: truths whose generating operator is DIAGONAL (hence perfectly normal, NG=1 at every
#            horizon).  Add a running-mean channel and see whether the pipeline reports NG(4)~2.
#            If yes, my ENSO "detection" carries no more physics than the choice of index.
# Control C : positive control - a genuinely non-normal 2x2 shear, to confirm the pipeline is sensitive.
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.linalg import solve_discrete_lyapunov
from enso_lib import ridge_fit, stats

RNG = np.random.default_rng(11)
N = 900
out = []


def ar1(phi, snr=1.0):
    x = np.zeros(N); x[0] = RNG.normal()
    for t in range(1, N):
        x[t] = phi * x[t - 1] + snr * RNG.normal()
    return x


def sh(v, k):
    return np.concatenate([np.zeros(k), v[:-k]]) if k > 0 else v


def trim(M, pad=6):
    return M[:, pad:N - pad]


def run(tag, Z):
    Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    A, S = ridge_fit(Z[:, :-1], Z[:, 1:])
    st = stats(A, S)
    rho = max(abs(np.linalg.eigvals(A)))
    defect = np.linalg.norm(A @ A.T - A.T @ A, "fro") / max(np.linalg.norm(A, "fro") ** 2, 1e-14)
    out.append("%-38s rho=%.4f  NG4=%5.2f  NG12=%5.2f  AG4=%.3f  ER12=%8.2f  commutator=%.3f"
               % (tag, rho, st["NG4"], st["NG12"], st["AG4"], st["ER12"], defect))


out.append("### Control A: ONE scalar AR(1) (phi=.95).  Truth is 1x1 => normal, true NG=1.00 everywhere.")
x = ar1(0.95)
m = np.convolve(x, np.ones(3) / 3, "same")
run("A0  [x]", trim(x[None, :]))
run("A1  [x, m]", trim(np.vstack([x, m])))
run("A2  [m, m(-1)]", trim(np.vstack([m, sh(m, 1)])))
run("A3  [m, m(-1), m(-2)]", trim(np.vstack([m, sh(m, 1), sh(m, 2)])))
run("A4  [x, m, m(-1)]", trim(np.vstack([x, m, sh(m, 1)])))
run("A5  [m, m(-1..-4)]", trim(np.vstack([m] + [sh(m, k) for k in (1, 2, 3, 4)])))
m12 = np.convolve(x, np.ones(12) / 12, "same")
run("A6  [m12, m12(-1..-2)] (12-mo mean)", trim(np.vstack([m12] + [sh(m12, k) for k in (1, 2)])))

out.append("")
out.append("### Control B: TWO independent scalar AR(1)s (block-diagonal truth, still normal).")
a, b = ar1(0.95), ar1(0.80)
b3 = np.convolve(b, np.ones(3) / 3, "same")
a3 = np.convolve(a, np.ones(3) / 3, "same")
run("B0  [a, b]", trim(np.vstack([a, b])))
run("B1  [a, b3]", trim(np.vstack([a, b3])))
run("B2  [a, b3, a3]", trim(np.vstack([a, b3, a3])))
run("B3  [a, a(-1), b, b3]", trim(np.vstack([a, sh(a, 1), b, b3])))
run("B4  [a, b] + both smoothed (d=4)", trim(np.vstack([a, b, a3, b3])))

out.append("")
out.append("### Control C: TRUE non-normal generator A=[[.98,g],[0,.80]] (true NG4 shown).")
for g in (0.0, 2.0, 6.0):
    Ac = np.array([[0.98, g], [0.0, 0.80]])
    P = solve_discrete_lyapunov(Ac, np.eye(2))
    z = np.zeros((2, N)); z[:, 0] = RNG.normal(size=2)
    L = np.linalg.cholesky(P + 1e-12 * np.eye(2))
    for t in range(1, N):
        z[:, t] = Ac @ z[:, t - 1] + L @ RNG.normal(size=2)
    true_ng4 = np.linalg.norm(np.linalg.matrix_power(Ac, 4), 2) / 0.98 ** 4
    out.append("  g=%.1f  TRUE NG4=%.2f  ->  fitted:" % (g, true_ng4))
    run("     fitted", trim(z))

out.append("")
out.append("### Reference: the real ENSO numbers b22 had to beat")
out.append("  SST4+ONI  d=5 T=900 : NG4=2.121 (H0 p95 1.549, p=0.000)")
out.append("  SST4+ONI+SOI d=6    : NG4=2.559 (H0 p95 1.510, p=0.000)")
out.append("  SST4 only d=4 T=900 : NG4=1.058 (H0 med 1.364, p=1.000)")
print("\n".join(out))
