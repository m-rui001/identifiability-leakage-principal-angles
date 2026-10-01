# B round 13: kill-or-keep B-CLAIM-24.  Is the observed non-normality (ER(12)=14-55) just SEASONALITY?
# Null generator H0s = "cyclostationary but NORMAL": month-specific operators A_m, each replaced by its
# same-spectrum NORMAL proxy, plus month-specific noise covariance.  Simulate at the SAME length,
# then run my ORIGINAL single-operator pipeline (which ignores seasonality) and compute ER/NG.
# If the seasonal-normal null reproduces the observed values => B-CLAIM-24 was seasonality, retract.
# Positive control: generate from the raw (non-normal) A_m and confirm the pipeline detects it.
import numpy as np, os
from scipy.linalg import schur

RNG = np.random.default_rng(202)
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
O = yearly12(os.path.join(D, "oni.data"))
S = yearly12(os.path.join(D, "soi.data"))
sc = lambda d: {k: float(v) for k, v in d.items()}
DAT = {"SST4+ONI+SOI_82_24": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)], (1982, 2024)),
       "SST4_1950_2024": ([{k: E[k][j] for k in E} for j in range(4)], (1950, 2024))}


def normal_proxy(A):
    T, Q = schur(A, output="real")
    d, Tn, i = A.shape[0], np.zeros_like(T), 0
    while i < d - 1:
        if abs(T[i + 1, i]) > 1e-12:
            Tn[i:i + 2, i:i + 2] = T[i:i + 2, i:i + 2]; i += 2
        else:
            Tn[i, i] = T[i, i]; i += 1
    return Q @ Tn @ Q.T


def ridge_fit(X0, X1, lam_rel=1e-2):
    d = X0.shape[0]
    C = X0 @ X0.T / X0.shape[1]
    lam = lam_rel * np.trace(C) / d
    A = np.linalg.solve(X0 @ X0.T + lam * np.eye(d), X0 @ X1.T).T
    R = X1 - A @ X0
    return A, R @ R.T / X0.shape[1]


def monthly_model(X, mon):
    """shrink month-specific fits toward the pooled fit (weight n_m/(n_m+d^2))."""
    d = X.shape[0]
    X0, X1 = X[:, :-1], X[:, 1:]
    Ap, Sp = ridge_fit(X0, X1)
    Am, Sm = [], []
    for m in range(1, 13):
        sel = mon[:-1] == m
        nm = int(sel.sum())
        if nm > d:
            a, s = ridge_fit(X0[:, sel], X1[:, sel], lam_rel=1e-1)
            w = nm / (nm + d * d)
            Am.append(w * a + (1 - w) * Ap); Sm.append(w * s + (1 - w) * Sp)
        else:
            Am.append(Ap); Sm.append(Sp)
    return Ap, Sp, Am, Sm


def stats(A, Sig, An=None):
    if An is None:
        An = normal_proxy(A)
    rho = max(abs(np.linalg.eigvals(A)))
    r = {}
    for n in (2, 4, 12):
        K, Kn = np.linalg.matrix_power(A, n), np.linalg.matrix_power(An, n)
        r["ER%d" % n] = np.trace(K @ Sig @ K.T) / max(np.trace(Kn @ Sig @ Kn.T), 1e-15)
        r["NG%d" % n] = np.linalg.norm(K, 2) / max(rho ** n, 1e-14)
    return r


def simulate(Am, Sm, mon, T, use_normal):
    d = len(Am) and Am[0].shape[0]
    L = [np.linalg.cholesky(s + 1e-10 * np.eye(d)) for s in Sm]
    x = np.zeros(d)
    X = np.zeros((d, T))
    for t in range(1, T):
        m = mon[t - 1] - 1
        A = normal_proxy(Am[m]) if use_normal else Am[m]
        x = A @ x + L[m] @ RNG.normal(size=d)
        X[:, t] = x
    X = X[:, 60:]
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


out = []
for name, (dicts, yrange) in DAT.items():
    idx = [k for k in [(y, m) for y in range(yrange[0], yrange[1] + 1) for m in range(1, 13)]
           if all(k in d for d in dicts)]
    X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    mon = np.array([k[1] for k in idx])
    T = X.shape[1]
    Ap, Sp, Am, Sm = monthly_model(X, mon)
    obs = stats(Ap, Sp)
    # how seasonal is the fitted monthly model itself?
    dev = np.mean([np.linalg.norm(Am[m] - Ap, "fro") / np.linalg.norm(Ap, "fro") for m in range(12)])
    out.append("### %s d=%d T=%d  pooled rho=%.4f  mean||A_m-A||_F/||A||_F=%.3f" %
               (name, Ap.shape[0], T, max(abs(np.linalg.eigvals(Ap))), dev))
    out.append("%8s %10s %10s %10s %10s %10s %10s" %
               ("stat", "observed", "H0norm50", "H0norm95", "p(seas-normal)", "H1s med", "H1s p"))
    ns = ["ER2", "ER4", "ER12", "NG12"]
    nullv = {k: [] for k in ns}
    h1v = {k: [] for k in ns}
    for _ in range(150):
        Xs = simulate(Am, Sm, mon, T, True)
        a, s = ridge_fit(Xs[:, :-1], Xs[:, 1:])
        st = stats(a, s)
        for k in ns:
            nullv[k].append(st[k])
    for _ in range(60):
        Xs = simulate(Am, Sm, mon, T, False)
        a, s = ridge_fit(Xs[:, :-1], Xs[:, 1:])
        st = stats(a, s)
        for k in ns:
            h1v[k].append(st[k])
    for k in ns:
        nv = np.array(nullv[k]); hv = np.array(h1v[k])
        out.append("%8s %10.3f %10.3f %10.3f %14.3f %10.3f %10.3f"
                   % (k, obs[k], np.median(nv), np.quantile(nv, .95), (nv >= obs[k]).mean(),
                      np.median(hv), (hv >= obs[k]).mean()))
    out.append("")
print("\n".join(out))
