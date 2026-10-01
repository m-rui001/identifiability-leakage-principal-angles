# B round 13d: b21 retired ER and left NG(12)=2.93 (p=0.007) standing, but the H1s arm showed
# the seasonal-NON-normal generator (fitted monthly A_m) reproduces the observed statistics.
# For d=6/T=516 each monthly fit has ~43 samples against d^2=36 parameters => A_m is noise-dominated,
# so that agreement may only say "noise operators reproduce noise statistics".
# Test: the SAME battery on windows where the monthly fits are well conditioned (n_m/d^2 >= 4).
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enso_lib import (DATA, yearly12, ersst, scalar, load, monthly_model, simulate,
                      stats, ridge_fit, normal_proxy, NS)

E = ersst(os.path.join(DATA, "ersst5.nino.mth.91-20.ascii"))
O = yearly12(os.path.join(DATA, "oni.data"))
S = yearly12(os.path.join(DATA, "soi.data"))
M = yearly12(os.path.join(DATA, "meiv2.data"))      # meiv2 is 12-values-per-year, NOT one row per month
sc = lambda d: {k: float(v) for k, v in d.items()}
SST = [{k: E[k][j] for k in E} for j in range(4)]

CASES = [("SST4                1950-2024", SST, 1950, 2024),
         ("SST4+SOI            1950-2024", SST + [sc(S)], 1951, 2024),
         ("SST4+ONI            1950-2024", SST + [sc(O)], 1950, 2024),
         ("SST4+MEI            1979-2024", SST + [sc(M)], 1979, 2024),
         ("SST4+ONI+SOI        1982-2024", SST + [sc(O), sc(S)], 1982, 2024)]

out = []
for name, dicts, y0, y1 in CASES:
    X, mon = load(dicts, y0, y1)
    T = X.shape[1]
    if T < 200:
        out.append("### %s SKIPPED (T=%d)" % (name, T))
        continue
    Ap, Sp, Am, Sm, nm = monthly_model(X, mon)
    obs = stats(Ap, Sp)
    dev = np.mean([np.linalg.norm(Am[m] - Ap, "fro") / np.linalg.norm(Ap, "fro") for m in range(12)])
    rho = max(abs(np.linalg.eigvals(Ap)))
    out.append("### %s  d=%d T=%d  n_m/min=%d (need d^2=%d)  rho=%.4f  seasonal dev=%.3f"
               % (name, Ap.shape[0], T, nm, Ap.shape[0] ** 2, rho, dev))
    rng = np.random.default_rng(202)
    null = {k: [] for k in NS}
    h1 = {k: [] for k in NS}
    for _ in range(150):
        Xs = simulate(Am, Sm, mon, T, True, rng)
        a, s = ridge_fit(Xs[:, :-1], Xs[:, 1:])
        st = stats(a, s)
        for k in NS:
            null[k].append(st[k])
    for _ in range(60):
        Xs = simulate(Am, Sm, mon, T, False, rng)
        a, s = ridge_fit(Xs[:, :-1], Xs[:, 1:])
        st = stats(a, s)
        for k in NS:
            h1[k].append(st[k])
    out.append("%8s %12s %10s %10s %6s %10s %6s" % ("stat", "observed", "H0 med", "H0 p95", "p", "H1s med", "p"))
    for k in NS:
        nv = np.array(null[k], float); hv = np.array(h1[k], float); ov = obs[k]
        out.append("%8s %12.3f %10.3f %10.3f %6.3f %10.3f %6.3f"
                   % (k, ov, np.median(nv), np.quantile(nv, .95), (nv >= ov).mean(),
                      np.median(hv), (hv >= ov).mean()))
    out.append("")
print("\n".join(out))
