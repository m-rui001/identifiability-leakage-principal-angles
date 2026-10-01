# B round 12b: B-HYP-1 on real NOAA/PSL data, done properly.
# Headline metric = forecast-error inflation  ER(n) = tr(A^n Sig A^nT) / tr(A_N^n Sig A_N^nT)
#   (A_N = normal proxy with the IDENTICAL spectrum, from the real Schur form)  -> well-conditioned, no logdet ratios.
# Also MI optimism, but reported with a floor on |I| to avoid the blow-up that ruined b16's CI.
# Controls: (pos) simulate from A_N (normal truth) -> pipeline must report ER ~ 1
#           (neg) simulate from A    (non-normal truth) -> pipeline must report ER > 1  (detectability)
# Bootstrap: moving blocks of 24 months, 300 reps, on the fitted operator.
import numpy as np, os
from scipy.linalg import schur

RNG = np.random.default_rng(17)
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


def anomify(d):
    bym = {}
    for (y, m), v in d.items():
        bym.setdefault(m, []).append(v)
    clim = {m: np.mean(v) for m, v in bym.items()}
    return {k: v - clim[k[1]] for k, v in d.items()}


E = ersst(os.path.join(D, "ersst5.nino.mth.91-20.ascii"))
S = yearly12(os.path.join(D, "soi.data"))
O = yearly12(os.path.join(D, "oni.data"))
N4 = anomify(yearly12(os.path.join(D, "nina4.data")))
M = yearly12(os.path.join(D, "meiv2.data"))
sc = lambda d: {k: float(v) for k, v in d.items()}

DAT = {
    "SST4": ([{k: E[k][j] for k in E} for j in range(4)], (1950, 2024)),
    "SST4+ONI+SOI": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)], (1950, 2024)),
    "ENSO8": ([{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S), sc(N4), sc(M)], (1982, 2024)),
}


def normal_proxy(A):
    T, Q = schur(A, output="real")
    d = A.shape[0]
    Tn = np.zeros_like(T)
    i = 0
    while i < d - 1:
        if abs(T[i + 1, i]) > 1e-12:
            Tn[i:i + 2, i:i + 2] = T[i:i + 2, i:i + 2]
            i += 2
        else:
            Tn[i, i] = T[i, i]
            i += 1
    return Q @ Tn @ Q.T


def fit(X, lam_rel=1e-3):
    n = X.shape[1]
    tr = int(0.7 * n)
    X0, X1 = X[:, :tr], X[:, 1:tr + 1]
    C = X0 @ X0.T / tr
    lam = lam_rel * np.trace(C) / C.shape[0]
    A = np.linalg.solve(X0 @ X0.T + lam * np.eye(C.shape[0]), X0 @ X1.T).T
    R = X1 - A @ X0
    Sig = R @ R.T / tr
    return A, C, Sig


def metrics(A, C, Sig, hs):
    rho = max(abs(np.linalg.eigvals(A)))
    An = normal_proxy(A)
    out = {"rho": rho}
    for n in hs:
        K = np.linalg.matrix_power(A, n)
        Kn = np.linalg.matrix_power(An, n)
        e1 = np.trace(K @ Sig @ K.T)
        e2 = np.trace(Kn @ Sig @ Kn.T)
        out[n] = (e1 / max(e2, 1e-12),
                  np.linalg.norm(K, 2) / max(rho ** n, 1e-14),
                  in_n(K, C, Sig, n, A), in_n(Kn, C, Sig, n, An))
    return out


def in_n(K, C, Sig, n, K0):
    d = K0.shape[0]
    M = np.zeros((d, d))
    Ki = np.eye(d)
    for i in range(n):
        M += Ki @ Sig @ Ki.T
        Ki = K0 @ Ki
    return 0.5 * np.linalg.slogdet(np.eye(d) + np.linalg.solve(M, K @ C @ K.T))[1]


def simulate(A, Sig, T, x0=0.0):
    d = A.shape[0]
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(d))
    x = np.zeros((d, 1))
    X = np.zeros((d, T))
    for t in range(1, T):
        x = A @ x + L @ RNG.normal(size=d).reshape(-1, 1)
        X[:, t:t + 1] = x
    return X[:, 100:]


out = []
hs = [1, 2, 4, 6, 12, 18, 24]
for name, (dicts, yrange) in DAT.items():
    idx = [k for k in [(y, m) for y in range(yrange[0], yrange[1] + 1) for m in range(1, 13)]
           if all(k in d for d in dicts)]
    X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    A, C, Sig = fit(X)
    m = metrics(A, C, Sig, hs)
    out.append("### %s  d=%d  n=%d  rho=%.4f  cond(eig)=%.3g" %
               (name, A.shape[0], X.shape[1], m["rho"], np.linalg.cond(np.linalg.eig(A)[1])))
    out.append("%4s %10s %10s %12s %12s %12s" % ("n", "ER(err ratio)", "NG", "I_true", "I_norm", "MI optim.%"))
    for n in hs:
        er, ng, i1, i2 = m[n]
        mi = 100.0 * (i2 - i1) / max(abs(i1), 0.02)
        out.append("%4d %10.3f %10.3f %12.5f %12.5f %12.2f" % (n, er, ng, i1, i2, mi))
    # bootstrap ER at n=6,12
    B = 24
    tr = int(0.7 * X.shape[1])
    er6, er12, ng12 = [], [], []
    for _ in range(300):
        st = RNG.integers(0, tr - B, size=tr // B)
        cols = np.concatenate([np.arange(s, s + B) for s in st])
        Xb = X[:, cols]
        Ab, Cb, Sb = fit(Xb)
        mb = metrics(Ab, Cb, Sb, [6, 12])
        er6.append(mb[6][0]); er12.append(mb[12][0]); ng12.append(mb[12][1])
    for lab, arr in (("ER(n=6)", er6), ("ER(n=12)", er12), ("NG(n=12)", ng12)):
        arr = np.array(arr)
        out.append("  bootstrap %s: median=%.3f  CI95=[%.3f, %.3f]  P(>1.05)=%.2f"
                   % (lab, np.median(arr), np.quantile(arr, .025), np.quantile(arr, .975), (arr > 1.05).mean()))
    # controls: simulate from the normal proxy (must give ER~1) and from A (must give ER>1)
    for lab, Ksim in (("control: normal truth A_N", normal_proxy(A)), ("control: non-normal truth A", A)):
        vals = []
        for _ in range(20):
            Xs = simulate(Ksim, Sig, 900)
            Xs = (Xs - Xs.mean(1, keepdims=True)) / Xs.std(1, keepdims=True)
            As, Cs, Ss = fit(Xs)
            vals.append(metrics(As, Cs, Ss, [12])[12][0])
        vals = np.array(vals)
        out.append("  %s -> fitted ER(n=12) median=%.3f  IQR=[%.3f, %.3f]  (detection power of the pipeline)"
                   % (lab, np.median(vals), np.quantile(vals, .25), np.quantile(vals, .75)))
    out.append("")
print("\n".join(out))
