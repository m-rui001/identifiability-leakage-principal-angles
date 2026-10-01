# B round 12: P1 -- REAL data (NOAA/PSL indices) test of B-HYP-1.
# Claim under test: on a real forced-damped system the Koopman/DMD operator is strongly NON-NORMAL,
# so the "normal-mode / spectral" picture of n-step uncertainty growth is OPTIMISTIC
# (under-predicts forecast error, over-predicts retained information) by >>5%, growing with horizon.
# Design: same eigenvalues, normal vs non-normal proxy -- build the normal proxy from the REAL Schur form
# (keep only the diagonal 1x1/2x2 blocks, which is normal and has the identical spectrum).
# Then % optimism = (I_n(normal proxy) - I_n(fitted))/I_n(fitted), and the forecast-error ratio.
# UQ: moving-block bootstrap (block=24 mo) on the fitted operator.
import numpy as np, os
from scipy.linalg import schur

RNG = np.random.default_rng(5)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def yearly12(path, y0col=0):
    """PSL 'YEAR v1..v12' files -> dict (year,month)->value, skipping -99.99."""
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
    """YR MON SST ANOM x4 -> dict (y,m)->[4 anomalies]."""
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
    """degC series -> monthly climatology anomaly."""
    bym = {}
    for (y, m), v in d.items():
        bym.setdefault(m, []).append(v)
    clim = {m: np.mean(v) for m, v in bym.items()}
    return {k: v - clim[k[1]] for k, v in d.items()}


def series(dicts, names, yrange):
    idx = [k for k in [(y, m) for y in range(yrange[0], yrange[1] + 1) for m in range(1, 13)]
           if all(k in d for d in dicts)]
    X = np.array([[dicts[j][k] for j in range(len(dicts))] for k in idx]).T
    return idx, X


E = ersst(os.path.join(D, "ersst5.nino.mth.91-20.ascii"))
S = yearly12(os.path.join(D, "soi.data"))
O = yearly12(os.path.join(D, "oni.data"))
N4 = anomify(yearly12(os.path.join(D, "nina4.data")))
M = yearly12(os.path.join(D, "meiv2.data"))


def scalar(d):
    return {k: (v if np.isscalar(v) else float(v)) for k, v in d.items()}


DATASETS = {}
DATASETS["SST4_1950_2024"] = ([{k: E[k][j] for k in E} for j in range(4)], (1950, 2024),
                             ["Nino1+2", "Nino3", "Nino4", "Nino3.4"])
DATASETS["SST4+ONI+SOi_1950_2024"] = ([{k: E[k][j] for k in E} for j in range(4)] + [scalar(O), scalar(S)],
                                      (1950, 2024), ["Nino1+2", "Nino3", "Nino4", "Nino3.4", "ONI", "SOI"])
DATASETS["ENSO8_1982_2024"] = ([{k: E[k][j] for k in E} for j in range(4)] + [scalar(O), scalar(S), scalar(N4), scalar(M)],
                               (1982, 2024), ["N12", "N3", "N4", "N34", "ONI", "SOI", "N4an", "MEI"])


def real_schur_normal_proxy(A):
    T, Q = schur(A, output='real')
    d = A.shape[0]
    Tn = np.zeros_like(T)
    i = 0
    while i < d - 1:
        if abs(T[i + 1, i]) > 1e-12:          # 2x2 rotation-scaling block (normal)
            Tn[i:i + 2, i:i + 2] = T[i:i + 2, i:i + 2]
            i += 2
        else:
            Tn[i, i] = T[i, i]
            i += 1
    return Q @ Tn @ Q.T


def in_n(K, C, Sig, n):
    dn = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = np.zeros((dn, dn))
    Ki = np.eye(dn)
    for i in range(n):
        M += Ki @ Sig @ Ki.T
        Ki = K @ Ki
    A = np.eye(dn) + np.linalg.solve(M, Kn @ C @ Kn.T)
    return 0.5 * np.linalg.slogdet(A)[1]


def analyze(A, C, Sig, hs):
    rho = max(abs(np.linalg.eigvals(A)))
    NG = [np.linalg.norm(np.linalg.matrix_power(A, n), 2) / max(rho ** n, 1e-12) for n in hs]
    An = real_schur_normal_proxy(A)
    I_true = [in_n(A, C, Sig, n) for n in hs]
    I_norm = [in_n(An, C, Sig, n) for n in hs]
    err_true = [np.trace(np.linalg.matrix_power(A, n) @ Sig @ np.linalg.matrix_power(A, n).T) for n in hs]
    err_norm = [np.trace(np.linalg.matrix_power(An, n) @ Sig @ np.linalg.matrix_power(An, n).T) for n in hs]
    err_spec = [rho ** (2 * n) * np.trace(Sig) for n in hs]
    return dict(rho=rho, NG=NG, I_true=I_true, I_norm=I_norm,
                err_true=err_true, err_norm=err_norm, err_spec=err_spec,
                cond=np.linalg.cond(np.linalg.eig(A)[1]) if np.linalg.cond(np.linalg.eig(A)[1]) < 1e12 else np.inf,
                wabs=float(np.linalg.eigvalsh((A + A.T) / 2)[-1]),
                nrm=float(np.linalg.norm(A, 2)))


out = []
hs = [1, 2, 3, 4, 6, 8, 12, 18, 24]
for name, (dicts, yrange, labels) in DATASETS.items():
    idx, X = series(dicts, labels, yrange)
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    n = X.shape[1]
    tr = int(0.7 * n)
    X0, X1 = X[:, :tr], X[:, 1:tr + 1]
    C = X0 @ X0.T / tr
    lam = 1e-3 * np.trace(C) / C.shape[0]
    A = np.linalg.solve(X0 @ X0.T + lam * np.eye(C.shape[0]), X0 @ X1.T).T
    R = X1 - A @ X0
    Sig = R @ R.T / tr
    r = analyze(A, C, Sig, hs)
    out.append("### %s  d=%d  n=%d (train %d)  rho=%.4f  ||A||_2=%.4f  num-abscissa=%.4f  eig-cond=%.3g" %
               (name, C.shape[0], n, tr, r["rho"], r["nrm"], r["wabs"], r["cond"]))
    out.append("%4s %10s %12s %12s %14s %14s %14s" % ("n", "NG", "I_true", "I_norm", "optim.%", "errT/errN", "errT/errSpec"))
    for i, hn in enumerate(hs):
        opt = 100.0 * (r["I_norm"][i] - r["I_true"][i]) / max(abs(r["I_true"][i]), 1e-12)
        out.append("%4d %10.3f %12.5f %12.5f %14.2f %14.3f %14.3f"
                   % (hn, r["NG"][i], r["I_true"][i], r["I_norm"][i], opt,
                      r["err_true"][i] / max(r["err_norm"][i], 1e-12),
                      r["err_true"][i] / max(r["err_spec"][i], 1e-12)))
    # block bootstrap of the headline optimism at n=12
    B = 24
    boots = []
    for _ in range(200):
        starts = RNG.integers(0, tr - B, size=tr // B)
        cols = np.concatenate([np.arange(s, s + B) for s in starts])
        Xb = X[:, cols]
        Xb0, Xb1 = Xb[:, :-1], Xb[:, 1:]
        Cb = Xb0 @ Xb0.T / Xb0.shape[1]
        Ab = np.linalg.solve(Xb0 @ Xb0.T + lam * np.eye(Cb.shape[0]), Xb0 @ Xb1.T).T
        Rb = Xb1 - Ab @ Xb0
        Sb = Rb @ Rb.T / Xb0.shape[1]
        rb = analyze(Ab, Cb, Sb, [12])
        boots.append(100.0 * (rb["I_norm"][0] - rb["I_true"][0]) / max(abs(rb["I_true"][0]), 1e-12))
    boots = np.array(boots)
    out.append("  block-bootstrap (B=24mo, 200 reps) optimism at n=12: mean=%.2f%%  CI95=[%.2f, %.2f]  P(>5%%)=%.2f"
               % (boots.mean(), np.quantile(boots, .025), np.quantile(boots, .975), (boots > 5).mean()))
    out.append("")
print("\n".join(out))
