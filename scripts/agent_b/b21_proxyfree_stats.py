# B round 13c: b20 showed ER(n)=tr(A^n Sig A^nT)/tr(An^n Sig An^nT) is a RATIO WITH AN UNSTABLE
# DENOMINATOR: dropping one of six variables swings it 0.76 -> 28.06, and the moving-block bootstrap
# median is 4.8 with 90% CI [0.86, 71.9]. Root cause: the normal-proxy denominator tr(Atilde^n Sig ...)
# can be near zero when the proxy's most amplified direction is orthogonal to Sig.
# So: re-run the whole battery with PROXY-FREE amplification statistics.
#   AG(n)  = ||A^n L||_F / (rho^n ||L||_F)         L=chol(Sig): noise-driven energy amplification
#   NG(n)  = ||A^n||_2   / rho^n                    operator (worst-case) transient growth
#   DR(n)  = tr(A^n Sig A^nT) / (tr(Sig) rho^{2n})   forecast-error variance ratio to MODAL decay
# DR/AG need no normal proxy at all => no small-denominator failure mode.
import numpy as np, os
from scipy.linalg import schur


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


NS = ["AG12", "NG12", "DR12", "AG4", "DR4", "ER12"]


def stats(A, Sig):
    d = A.shape[0]
    rho = max(abs(np.linalg.eigvals(A)))
    Sig = np.asarray(Sig, dtype=float)
    w, V = np.linalg.eigh(Sig)
    L = V @ np.diag(np.sqrt(np.maximum(w, 0)))          # symmetric Sig^{1/2}, always real
    r = {"ER12": np.nan}
    An = normal_proxy(A)
    for n in (4, 12):
        K = np.linalg.matrix_power(A, n)
        KL = K @ L
        base = np.linalg.norm(L, "fro") * rho ** n
        r["AG%d" % n] = np.linalg.norm(KL, "fro") / max(base, 1e-14)
        r["DR%d" % n] = np.trace(K @ Sig @ K.T) / max(np.trace(Sig) * rho ** (2 * n), 1e-14)
        r["NG%d" % n] = np.linalg.norm(K, 2) / max(rho ** n, 1e-14)
        if n == 12:
            Kn = np.linalg.matrix_power(An, n)
            r["ER12"] = np.trace(K @ Sig @ K.T) / max(np.trace(Kn @ Sig @ Kn.T), 1e-15)
    return r


def load(dicts, y0, y1):
    idx = [k for k in [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]
           if all(k in d for d in dicts)]
    X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return X, np.array([k[1] for k in idx])


def monthly_model(X, mon):
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


RNG = np.random.default_rng(202)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
E = ersst(os.path.join(D, "ersst5.nino.mth.91-20.ascii"))
O = yearly12(os.path.join(D, "oni.data"))
S = yearly12(os.path.join(D, "soi.data"))
sc = lambda d: {k: float(v) for k, v in d.items()}
LBL = ["Nino1+2", "Nino3", "Nino3.4", "Nino4", "ONI", "SOI"]
full = [{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)]

X, mon = load(full, 1982, 2024)
Ap, Sp, Am, Sm = monthly_model(X, mon)
obs = stats(Ap, Sp)
out = ["### d=6 1982-2024  rho=%.4f  (b19 seasonal-normal null, 150 reps)" % max(abs(np.linalg.eigvals(Ap)))]


def simulate(Am, Sm, mon, T, use_normal):
    d = Am[0].shape[0]
    Lm = []
    for s in Sm:
        w, V = np.linalg.eigh(np.asarray(s, dtype=float) + 1e-10 * np.eye(d))
        Lm.append(V @ np.diag(np.sqrt(np.maximum(w, 0))))
    x = np.zeros(d); Xs = np.zeros((d, T))
    for t in range(1, T):
        m = mon[t - 1] - 1
        A = normal_proxy(Am[m]) if use_normal else Am[m]
        x = A @ x + Lm[m] @ RNG.normal(size=d)
        Xs[:, t] = x
    Xs = Xs[:, 60:]
    return (Xs - Xs.mean(1, keepdims=True)) / Xs.std(1, keepdims=True)


def battery(Xgen, mon, label):
    a, s = ridge_fit(Xgen[:, :-1], Xgen[:, 1:])
    return stats(a, s)


null = {k: [] for k in NS}
h1 = {k: [] for k in NS}
for _ in range(150):
    Xs = simulate(Am, Sm, mon, X.shape[1], True)
    st = battery(Xs, mon[:Xs.shape[1]], "")
    for k in NS:
        null[k].append(st[k])
for _ in range(60):
    Xs = simulate(Am, Sm, mon, X.shape[1], False)
    st = battery(Xs, mon[:Xs.shape[1]], "")
    for k in NS:
        h1[k].append(st[k])
out.append("%8s %12s %10s %10s %8s %10s %8s" % ("stat", "observed", "H0 med", "H0 p95", "p", "H1s med", "p"))
for k in NS:
    nv = np.array(null[k], dtype=float); hv = np.array(h1[k], dtype=float)
    ov = obs[k]
    out.append("%8s %12.3f %10.3f %10.3f %8.3f %10.3f %8.3f" %
               (k, ov, np.nanmedian(nv), np.nanquantile(nv, .95), (nv >= ov).mean(),
                np.nanmedian(hv), (hv >= ov).mean()))

out.append("")
out.append("### leave-one-variable-out (d=5, proxy-free stats)")
out.append("%10s %8s %10s %10s %10s %10s" % ("drop", "rho", "AG12", "NG12", "DR12", "ER12"))
for j in range(6):
    keep = [i for i in range(6) if i != j]
    Xd, md = load([full[i] for i in keep], 1982, 2024)
    a, s = ridge_fit(Xd[:, :-1], Xd[:, 1:])
    st = stats(a, s)
    out.append("%10s %8.4f %10.3f %10.3f %10.3f %10.2f" %
               (LBL[j], max(abs(np.linalg.eigvals(a))), st["AG12"], st["NG12"], st["DR12"], st["ER12"]))

out.append("")
out.append("### epoch split (d=6, T=258 each)")
out.append("%12s %8s %10s %10s %10s %10s" % ("epoch", "rho", "AG12", "NG12", "DR12", "ER12"))
for lab, a, b in [("1982-2003", 0, 258), ("2003-2024", 258, 516)]:
    sub = X[:, a:b]
    sub = (sub - sub.mean(1, keepdims=True)) / sub.std(1, keepdims=True)
    aa, ss = ridge_fit(sub[:, :-1], sub[:, 1:])
    st = stats(aa, ss)
    out.append("%12s %8.4f %10.3f %10.3f %10.3f %10.2f" %
               (lab, max(abs(np.linalg.eigvals(aa))), st["AG12"], st["NG12"], st["DR12"], st["ER12"]))

out.append("")
out.append("### moving-block bootstrap of the OBSERVED statistic (block=36 mo, 200 reps)")
T = X.shape[1]
blk = int(np.ceil(T / 36))
bo = {k: [] for k in NS}
for _ in range(200):
    starts = RNG.integers(0, T - 36 + 1, size=blk)
    Xb = np.concatenate([X[:, s:s + 36] for s in starts], axis=1)[:, :T]
    Xb = (Xb - Xb.mean(1, keepdims=True)) / Xb.std(1, keepdims=True)
    ab, sb = ridge_fit(Xb[:, :-1], Xb[:, 1:])
    st = stats(ab, sb)
    for k in NS:
        bo[k].append(st[k])
out.append("%8s %12s %10s %10s %10s %12s" % ("stat", "point", "med", "p05", "p95", "P(>= H0 p95)"))
for k in NS:
    bv = np.array(bo[k], dtype=float)
    p95 = np.nanquantile(np.array(null[k], dtype=float), .95)
    out.append("%8s %12.3f %10.3f %10.3f %10.3f %12.3f" %
               (k, obs[k], np.nanmedian(bv), np.nanquantile(bv, .05), np.nanquantile(bv, .95),
                (np.array(null[k], dtype=float) < np.nanmedian(bv)).mean()))
print("\n".join(out))
