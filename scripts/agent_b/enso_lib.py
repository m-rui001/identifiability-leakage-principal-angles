# Shared ENSO-index helpers (B round 13+).  Extracted from b21 so the longer-window
# datasets in b22 use the identical pipeline.
import numpy as np, os
from scipy.linalg import schur

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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


def scalar(path):
    out = {}
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if len(t) < 2:
            continue
        try:
            y, m = int(float(t[0])), int(float(t[1]))
        except ValueError:
            continue
        if 1 <= m <= 12:
            out[(y, m)] = float(t[2])
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


def sqrtm_psd(S):
    w, V = np.linalg.eigh(np.asarray(S, dtype=float))
    return V @ np.diag(np.sqrt(np.maximum(w, 0)))


def stats(A, Sig):
    """proxy-free amplification statistics; ER is kept only as the known-bad control."""
    rho = max(abs(np.linalg.eigvals(A)))
    L = sqrtm_psd(Sig)
    nL = np.linalg.norm(L, "fro")
    r = {}
    for n in (4, 12):
        K = np.linalg.matrix_power(A, n)
        r["AG%d" % n] = np.linalg.norm(K @ L, "fro") / max(nL * rho ** n, 1e-14)
        r["DR%d" % n] = np.trace(K @ Sig @ K.T) / max(nL ** 2 * rho ** (2 * n), 1e-14)
        r["NG%d" % n] = np.linalg.norm(K, 2) / max(rho ** n, 1e-14)
    An = normal_proxy(A)
    K12, Kn12 = np.linalg.matrix_power(A, 12), np.linalg.matrix_power(An, 12)
    r["ER12"] = np.trace(K12 @ Sig @ K12.T) / max(np.trace(Kn12 @ Sig @ Kn12.T), 1e-15)
    return r


NS = ["NG4", "NG12", "AG4", "AG12", "DR4", "DR12", "ER12"]


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
    Am, Sm, nm_list = [], [], []
    for m in range(1, 13):
        sel = mon[:-1] == m
        nm = int(sel.sum())
        nm_list.append(nm)
        if nm > d:
            a, s = ridge_fit(X0[:, sel], X1[:, sel], lam_rel=1e-1)
            w = nm / (nm + d * d)
            Am.append(w * a + (1 - w) * Ap); Sm.append(w * s + (1 - w) * Sp)
        else:
            Am.append(Ap); Sm.append(Sp)
    return Ap, Sp, Am, Sm, min(nm_list)


def simulate(Am, Sm, mon, T, use_normal, rng):
    d = Am[0].shape[0]
    Lm = [sqrtm_psd(s) for s in Sm]
    x = np.zeros(d); Xs = np.zeros((d, T))
    for t in range(1, T):
        m = mon[t - 1] - 1
        A = normal_proxy(Am[m]) if use_normal else Am[m]
        x = A @ x + Lm[m] @ rng.normal(size=d)
        Xs[:, t] = x
    Xs = Xs[:, 60:]
    return (Xs - Xs.mean(1, keepdims=True)) / Xs.std(1, keepdims=True)
