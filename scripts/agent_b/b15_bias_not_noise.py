# B round 11b: is the sampling remainder a FLUCTUATION or a one-sided BIAS?
# Since E[G'G] = n * Gop exactly, lambda_min(Gop) is the "deterministic prediction". Test:
#   (a) signed tail of eps  -> if median < 0 and the negative tail is heavy, it is a bias, not noise.
#   (b) Jensen gap  E[lambda_min(G'G)/n] / lambda_min(Gop)  vs n   (expect < 1, decaying as 1/n)
#   (c) second-order prediction   bias ~= -E[ ||P_perp Delta v1||^2 / (lam_2 - lam_1) ] / n^2 ... measure it
#       using the actual perturbation Delta = G'G - n Gop and the eigenvector v1 of Gop.
import numpy as np

rng = np.random.default_rng(31)


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
out.append("=== (a) signed tail of eps = sig_min(G)/sqrt(n lam_min) - 1,  p+q=24 (p=q=12), 400 trials ===")
for n in (200, 600, 1800):
    es = []
    for _ in range(400):
        E, F, Gop = draw(d, 12, 12, alpha)
        lm = max(np.linalg.eigvalsh(Gop)[0], 1e-15)
        U = rng.normal(size=(d, n))
        G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
        es.append(np.linalg.svd(G, compute_uv=False)[-1] / np.sqrt(n * lm) - 1.0)
    es = np.array(es)
    qs = [np.quantile(es, x) for x in (0.0025, 0.01, 0.05, 0.5, 0.95, 0.99, 0.9975)]
    out.append("n=%4d  q0.0025=%+.4f q0.01=%+.4f q0.05=%+.4f med=%+.4f q0.95=%+.4f q0.99=%+.4f q0.9975=%+.4f  mean=%+.4f sd=%.4f"
               % (n, qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], es.mean(), es.std()))
    out.append("        -> negative tail magnitude / positive tail magnitude = %.2f   (=> one-sided if >>1)"
               % (abs(qs[0] - qs[3]) / abs(qs[6] - qs[3])))
out.append("")

out.append("=== (b) Jensen gap: E[lam_min(G'G)/n] / lam_min(Gop)  (p+q=24) ===")
for n in (100, 200, 600, 1800, 5400):
    g = []
    for _ in range(200):
        E, F, Gop = draw(d, 12, 12, alpha)
        lm = max(np.linalg.eigvalsh(Gop)[0], 1e-15)
        U = rng.normal(size=(d, n))
        G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
        g.append(np.linalg.eigvalsh(G.T @ G)[0] / n / lm)
    out.append("n=%5d  ratio = %.5f   (1-ratio = %.5f)   vs 1/n = %.5f   vs 1/sqrt(n) = %.5f"
               % (n, np.mean(g), 1 - np.mean(g), 1.0 / n, 1 / np.sqrt(n)))
out.append("")

out.append("=== (c) second-order perturbation prediction of the bias, using the ACTUAL Delta = G'G - n Gop ===")
out.append("    bias2 = -E[ sum_{k>=2} <v1,Delta vk>^2 / (lam_k - lam_1) ] / n   compared with measured (1-ratio)*lam_min")
for n in (200, 600, 1800):
    b2, meas, gap = [], [], []
    for _ in range(200):
        E, F, Gop = draw(d, 12, 12, alpha)
        w, V = np.linalg.eigh(Gop)
        lm = max(w[0], 1e-15)
        U = rng.normal(size=(d, n))
        G = np.column_stack([(M @ U).reshape(-1) for M in E + F])
        Dm = G.T @ G - n * Gop
        c = V.T @ Dm @ V[:, 0]
        denom = w[1:] - w[0]
        b2.append(-np.sum(c[1:] ** 2 / denom) / n)
        lmg = np.linalg.eigvalsh(G.T @ G)[0]
        meas.append(lmg - n * lm)
        gap.append(denom[0])
    out.append("n=%4d  measured mean shift = %+.4e   second-order pred = %+.4e   ratio = %.3f   (lam_2-lam_1)/n=%.4f"
               % (n, np.mean(meas), np.mean(b2), np.mean(meas) / np.mean(b2), np.mean(gap) / n))
print("\n".join(out))
