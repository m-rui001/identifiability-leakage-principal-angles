# B round 11: G-alpha -- sharpen the sampling remainder of Theorem 4.2.
# Two candidate predictors for the relative deviation eps of sigma_min(G) from sqrt(n)*sigma_min(Gop^{1/2}):
#   (P1) kappa * r * sqrt((p+q)/n)            <- net over the coefficient sphere (dimension p+q)
#   (P2) kappa * r * sqrt((D_eff)/n) , D_eff = min(p+q, d^2)  <- span dimension of the dictionary
#   (P3) kappa * r * sqrt((p+q + 2 log(1/eta))/n)  <- subexponential net bound with failure prob eta
# Q: which one tracks the data as p+q grows, and where does kappa stop being constant (breakdown)?
import numpy as np

rng = np.random.default_rng(23)


def orth(A):
    return np.linalg.qr(A)[0]


def draw(d, p, q, alpha):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    F = [np.cos(alpha) * a[:, j].reshape(d, d) + np.sin(alpha) * b[:, j].reshape(d, d) for j in range(q)]
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(p)] for i in range(p)])
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(q)] for i in range(p)])
    Gop = np.zeros((p + q, p + q))
    Gop[:p, :p] = Lam
    Gop[:p, p:] = np.cos(alpha) * H
    Gop[p:, :p] = np.cos(alpha) * H.T
    Gop[p:, p:] = np.eye(q)
    return E, F, Gop


out = []
d, alpha = 10, np.deg2rad(5.0)
out.append("=== kappa table: |eps| / ( r * sqrt(X/n) ) for X = p+q, D_eff, p+q+2log(1/eta)  (d=10, alpha=5deg, 8 trials) ===")
out.append("%5s %6s %9s %10s %10s %10s %10s %10s"
           % ("p+q", "n", "|eps|", "r", "k(P1)", "k(P2)", "k(P3)", "lam_min(Gop)"))
eta = 0.05
for (p, q) in ((1, 1), (2, 2), (4, 2), (6, 6), (12, 12), (24, 24), (30, 20)):
    for n in (200, 600, 1800):
        eps, rr, lms = [], [], []
        for _ in range(8):
            E, F, Gop = draw(d, p, q, alpha)
            w, V = np.linalg.eigh(Gop)
            lm = max(w[0], 1e-15)
            v = V[:, 0]
            Mx = sum(v[i] * E[i] for i in range(p)) + sum(v[p + j] * F[j] for j in range(q))
            rr.append(np.linalg.norm(Mx @ Mx.T, "fro") / np.linalg.norm(Mx, "fro") ** 2)
            U = rng.normal(size=(d, n))
            G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
            sG = np.linalg.svd(G, compute_uv=False)[-1]
            eps.append(sG / np.sqrt(n * lm) - 1.0)
            lms.append(lm)
        e = np.mean(np.abs(eps))
        r = np.mean(rr)
        D = min(p + q, d * d)
        k1 = e / (r * np.sqrt((p + q) / n))
        k2 = e / (r * np.sqrt(D / n))
        k3 = e / (r * np.sqrt((p + q + 2 * np.log(1 / eta)) / n))
        out.append("%5d %6d %9.5f %10.4f %10.4f %10.4f %10.4f %10.3e"
                   % (p + q, n, e, r, k1, k2, k3, np.mean(lms)))
out.append("")

out.append("=== breakdown: does alpha-flatness of |eps| survive large p+q?  (d=10, n=600, eps at alpha=1 vs 30 deg) ===")
for (p, q) in ((4, 2), (12, 12), (30, 20)):
    row = []
    for ad in (1.0, 30.0):
        a = np.deg2rad(ad)
        es = []
        for _ in range(6):
            E, F, Gop = draw(d, p, q, a)
            lm = max(np.linalg.eigvalsh(Gop)[0], 1e-15)
            U = rng.normal(size=(d, 600))
            G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
            es.append(np.linalg.svd(G, compute_uv=False)[-1] / np.sqrt(600 * lm) - 1.0)
        row.append(np.mean(np.abs(es)))
    out.append("p+q=%3d  |eps|(a=1)=%.4f  |eps|(a=30)=%.4f  ratio=%.2f" % (p + q, row[0], row[1], row[0] / row[1]))
out.append("")

out.append("=== high-dim corner: d=6 (d^2=36) with p+q up to 34, n in {200,600}: is the predictor r*sqrt((p+q)/n) still OK? ===")
for n in (200, 600):
    for (p, q) in ((2, 2), (6, 6), (12, 10), (18, 16)):
        es = []
        for _ in range(6):
            E, F, Gop = draw(6, p, q, alpha)
            lm = max(np.linalg.eigvalsh(Gop)[0], 1e-15)
            U = rng.normal(size=(6, n))
            G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
            sG = np.linalg.svd(G, compute_uv=False)[-1]
            es.append(sG / np.sqrt(n * lm) - 1.0)
        pr = np.sqrt((p + q) / n)
        out.append("d=6 n=%4d p+q=%3d  |eps|=%.4f  sqrt((p+q)/n)=%.4f  kappa=%.3f  worst=%.4f"
                   % (n, p + q, np.mean(np.abs(es)), pr, np.mean(np.abs(es)) / pr, np.max(np.abs(es))))
out.append("")

out.append("=== eta-calibration: empirical tail of |eps| vs the subexponential predictor (p+q=24,d=10,n=600) ===")
tail = []
for _ in range(400):
    E, F, Gop = draw(d, 12, 12, alpha)
    lm = max(np.linalg.eigvalsh(Gop)[0], 1e-15)
    U = rng.normal(size=(d, 600))
    G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
    tail.append(np.linalg.svd(G, compute_uv=False)[-1] / np.sqrt(600 * lm) - 1.0)
tail = np.array(tail)
for nom in (0.50, 0.90, 0.95, 0.99, 0.9975):
    out.append("quantile %.4f of eps = %+.5f   |  sqrt((p+q+2log(1/eta))/n)=%.5f  -> implied kappa=%.3f"
               % (nom, np.quantile(tail, nom), np.sqrt((24 + 2 * np.log(1 / max(1 - nom, 1e-6))) / 600),
                  np.abs(np.quantile(tail, nom)) / np.sqrt((24 + 2 * np.log(1 / max(1 - nom, 1e-6))) / 600)))
print("\n".join(out))
