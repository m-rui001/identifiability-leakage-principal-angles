# B round 13b: after b19 killed the seasonality confound (p=0.000 for ER12 at d=6),
# ask WHERE the observed inflation lives: spectrum, epoch stability, block-bootstrap CI.
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


def ER_NG(A, Sig, N=12):
    An = normal_proxy(A)
    rho = max(abs(np.linalg.eigvals(A)))
    K, Kn = np.linalg.matrix_power(A, N), np.linalg.matrix_power(An, N)
    er = np.trace(K @ Sig @ K.T) / max(np.trace(Kn @ Sig @ Kn.T), 1e-15)
    ng = np.linalg.norm(K, 2) / max(rho ** N, 1e-14)
    return er, ng


D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
E = ersst(os.path.join(D, "ersst5.nino.mth.91-20.ascii"))
O = yearly12(os.path.join(D, "oni.data"))
S = yearly12(os.path.join(D, "soi.data"))
sc = lambda d: {k: float(v) for k, v in d.items()}
dicts = [{k: E[k][j] for k in E} for j in range(4)] + [sc(O), sc(S)]
idx = [k for k in [(y, m) for y in range(1982, 2025) for m in range(1, 13)] if all(k in d for d in dicts)]
X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
lbl = ["Nino1+2", "Nino3", "Nino3.4", "Nino4", "ONI", "SOI"]

out = ["### 1. pooled d=6 spectrum (d=6, T=%d)" % X.shape[1]]
A, Sig = ridge_fit(X[:, :-1], X[:, 1:])
w = np.linalg.eigvals(A)
order = np.argsort(-np.abs(w))
er, ng = ER_NG(A, Sig)
out.append("rho=%.4f  max Re lambda=%.4f  ||A||_2=%.3f  numerical abscissa(max eig sym)=%.3f" %
           (max(abs(w)), max(w.real), np.linalg.norm(A, 2), max(np.linalg.eigvals((A + A.T) / 2).real)))
out.append("ER(12)=%.2f  NG(12)=%.2f   [normal proxy would give ER=1 exactly]" % (er, ng))
for i in order:
    out.append("  |lam|=%.4f  lam=%.4f%+.4fj  period~%.1f yr" %
               (abs(w[i]), w[i].real, w[i].imag, (2 * np.pi / abs(np.angle(w[i] + 0j))) / 12 if abs(w[i]) < 1 and abs(np.angle(w[i])) > 1e-9 else float("nan")))

out.append("")
out.append("### 2. epoch split (re-fit independently, same pipeline)")
for j, (a, b) in enumerate([(0, 258), (258, 516)]):
    sub = X[:, a:b]
    sub = (sub - sub.mean(1, keepdims=True)) / sub.std(1, keepdims=True)
    Aa, Ss = ridge_fit(sub[:, :-1], sub[:, 1:])
    e, g = ER_NG(Aa, Ss)
    out.append("  %s (%s-%s, T=%d): rho=%.4f  ER(12)=%.2f  NG(12)=%.2f" %
               ("first half" if j == 0 else "second half", idx[a][0], idx[b - 1][0], b - a,
                max(abs(np.linalg.eigvals(Aa))), e, g))

out.append("")
out.append("### 3. leave-one-variable-out (which variable carries the effect?)")
for drop in range(6):
    keep = [j for j in range(6) if j != drop]
    Xd = X[keep]
    Xd = (Xd - Xd.mean(1, keepdims=True)) / Xd.std(1, keepdims=True)
    Ad, Sd = ridge_fit(Xd[:, :-1], Xd[:, 1:])
    e, g = ER_NG(Ad, Sd)
    out.append("  drop %-8s d=5: rho=%.4f  ER(12)=%8.2f  NG(12)=%.2f" % (lbl[drop], max(abs(np.linalg.eigvals(Ad))), e, g))

out.append("")
out.append("### 4. moving-block bootstrap of the OBSERVED statistic (block=36 months, 200 reps)")
RNG = np.random.default_rng(7)
T = X.shape[1]
blk = int(np.ceil(T / 36))
ers, ngs = [], []
for _ in range(200):
    starts = RNG.integers(0, T - 36 + 1, size=blk)
    Xs = np.concatenate([X[:, s:s + 36] for s in starts], axis=1)[:, :T]
    Xs = (Xs - Xs.mean(1, keepdims=True)) / Xs.std(1, keepdims=True)
    Ab, Sb = ridge_fit(Xs[:, :-1], Xs[:, 1:])
    e, g = ER_NG(Ab, Sb)
    ers.append(e); ngs.append(g)
ers = np.array(ers); ngs = np.array(ngs)
out.append("  ER(12): point %.2f   CI 5-95%% [%.2f, %.2f]   median %.2f" %
           (er, np.quantile(ers, .05), np.quantile(ers, .95), np.median(ers)))
out.append("  NG(12): point %.2f   CI 5-95%% [%.2f, %.2f]   median %.2f" %
           (ng, np.quantile(ngs, .05), np.quantile(ngs, .95), np.median(ngs)))
out.append("  bootstrap resampling destroys month order => block=36 keeps 3 yr of seasonality intact;")
out.append("  if the CI still excludes the b19 seasonal-normal 95%% (1.92) the effect is not one epoch.")
print("\n".join(out))
