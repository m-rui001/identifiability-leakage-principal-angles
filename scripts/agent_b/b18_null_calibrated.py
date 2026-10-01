# B round 12c: calibrate the non-normality statistics against a SIZE-MATCHED parametric null.
# b17's controls showed the pipeline reports ER(n=12) ~ 1.5 even when the truth is NORMAL ->
# ER/NG are inflated by estimation noise, and b17's null used T=900 while ENSO8 has only 516 months.
# Here: H0 = normal dynamics (A_N fitted from real data, same Sig, SAME sample size), 300 reps.
# Report observed statistic vs null 50th / 95th percentile  => honest p-value, no hand-waving.
# H1 (power) = simulate from the fitted non-normal A at the same size.
import numpy as np, os
from scipy.linalg import schur

RNG = np.random.default_rng(101)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def yearly12(path):
    out = {}
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if len(t) < 13:
            continue
        try:
            y = int(float(t[0]))
        except ValueError:
            continue
        if y < 1000:
            continue
        for m in range(12):
            v = float(t[1 + m])
            if v > -90:
                out[(y, m + 1)] = v
    return out


def ersst(path):
    out = {}
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if len(t) < 9:
            continue
        try:
            y, m = int(t[0]), int(t[1])
        except ValueError:
            continue
        if 1 <= m <= 12:
            out[(y, m)] = np.array([float(t[3]), float(t[5]), float(t[7]), float(t[9])])
    return out


E = ersst(os.path.join(D, "ersst5.nino.mth.91-20.ascii"))
S = yearly12(os.path.join(D, "soi.data"))
O = yearly12(os.path.join(D, "oni.data"))
sc = lambda d: {k: float(v) for k, v in d.items()}
DAT = {"SST4": ([{k: E[k][j] for k in E} for j in range(4)], (1950, 2024)),
       "SST4+ONI+SOI": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)], (1950, 2024)),
       "ENSO6b": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(M := yearly12(os.path.join(D, "meiv2.data")))], (1982, 2024)),
       # isolation arm: SAME window/size as ENSO6b but WITHOUT the interpolated MEI variable
       "SST4+ONI+SOI_82_24": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)], (1982, 2024)),
       # and the reverse: MEI in the long window
       "SST4+ONI+MEI_79_24": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(M)], (1979, 2024))}


def normal_proxy(A):
    T, Q = schur(A, output="real")
    d, Tn, i = A.shape[0], np.zeros_like(T), 0
    while i < d - 1:
        if abs(T[i + 1, i]) > 1e-12:
            Tn[i:i + 2, i:i + 2] = T[i:i + 2, i:i + 2]; i += 2
        else:
            Tn[i, i] = T[i, i]; i += 1
    return Q @ Tn @ Q.T


def fit(X):
    tr = int(0.7 * X.shape[1])
    X0, X1 = X[:, :tr], X[:, 1:tr + 1]
    C = X0 @ X0.T / tr
    lam = 1e-3 * np.trace(C) / C.shape[0]
    A = np.linalg.solve(X0 @ X0.T + lam * np.eye(C.shape[0]), X0 @ X1.T).T
    R = X1 - A @ X0
    return A, C, R @ R.T / tr


def stats(A, Sig):
    rho = max(abs(np.linalg.eigvals(A)))
    An = normal_proxy(A)
    r = {}
    for n in (2, 4, 12):
        K, Kn = np.linalg.matrix_power(A, n), np.linalg.matrix_power(An, n)
        r["ER%d" % n] = np.trace(K @ Sig @ K.T) / max(np.trace(Kn @ Sig @ Kn.T), 1e-12)
        r["NG%d" % n] = np.linalg.norm(K, 2) / max(rho ** n, 1e-14)
    r["NGoverERN"] = r["NG12"]
    return r


def simulate(K, Sig, T):
    d = K.shape[0]
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(d))
    X = np.zeros((d, T))
    x = np.zeros((d, 1))
    for t in range(1, T):
        x = K @ x + L @ RNG.normal(size=d).reshape(-1, 1)
        X[:, t] = x[:, 0]
    X = X[:, 50:]
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


out = []
KEYS = ["ER2", "ER4", "ER12", "NG12"]
for name, (dicts, yrange) in DAT.items():
    idx = [k for k in [(y, m) for y in range(yrange[0], yrange[1] + 1) for m in range(1, 13)]
           if all(k in d for d in dicts)]
    X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    T = X.shape[1]
    A, C, Sig = fit(X)
    obs = stats(A, Sig)
    An = normal_proxy(A)
    null, alt = [], []
    for _ in range(300):
        As, Cs, Ss = fit(simulate(An, Sig, T))
        null.append(stats(As, Ss))
    for _ in range(100):
        As, Cs, Ss = fit(simulate(A, Sig, T))
        alt.append(stats(As, Ss))
    out.append("### %s  d=%d  T=%d (train %d)  observed: %s" %
               (name, A.shape[0], T, int(0.7 * T), "  ".join("%s=%.3f" % (k, obs[k]) for k in KEYS)))
    out.append("%8s %10s %10s %10s %10s %10s" % ("stat", "observed", "null50", "null95", "p(null>=obs)", "H1 med"))
    for k in KEYS:
        nv = np.array([x[k] for x in null])
        av = np.array([x[k] for x in alt])
        out.append("%8s %10.3f %10.3f %10.3f %10.3f %10.3f"
                   % (k, obs[k], np.median(nv), np.quantile(nv, .95), (nv >= obs[k]).mean(), np.median(av)))
    out.append("")
print("\n".join(out))
