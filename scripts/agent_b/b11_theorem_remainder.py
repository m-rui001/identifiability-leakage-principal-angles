# B round 10: turn Prop 4.1 / Lemma A into theorems with EXPLICIT remainders.
# Q0: is the analytic operator Gram (Lam_E, cos a * H, I_q) exactly the true Frobenius Gram?
# Q1: identity lambda_min(Gop) = sin^2 a / (1 + cos^2 a * ||C||_2^2),  C = Lam_E^{-1} H ?
#     (this replaces my guessed 1/rho_E^2 in 17.7; ||C||_2^2 is the frame-theoretic quantity)
# Q2: sample remainder eps_n = sigma_min(G)/(sqrt n sigma_min(Gop^{1/2})) - 1: alpha-FREE? ~ sqrt((p+q)/n)?
#     my naive operator-norm bound gave O(d/(sqrt n sin^2 a)) -> catastrophic as a->0;
#     the quadratic-form (per-direction) bound says alpha-free. numerics decide.
# Q3: actual constant multiplying sqrt((p+q)/n); Q4: mechanism ratio ||MM^T||_F/||M||_F^2.
import numpy as np

rng = np.random.default_rng(7)


def orth(A):
    return np.linalg.qr(A)[0]


def prior_frame(d, p):
    """A's d10 verbatim: E_i = N(0,1)^{dxd}/sqrt(d)  =>  rho_E = ||E_i||_F ~ sqrt(d)."""
    return [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]


def build_F_at_angle(QE, q, alpha, d):
    """A's d10 verbatim (QE is d^2 x p orthonormal, so a,b are Frobenius-orthonormal matrices)."""
    a = orth(QE @ orth(rng.normal(size=(QE.shape[1], q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    ca, sa = np.cos(alpha), np.sin(alpha)
    return [a[:, j].reshape(d, d) for j in range(q)], [b[:, j].reshape(d, d) for j in range(q)], \
           [ca * a[:, j].reshape(d, d) + sa * b[:, j].reshape(d, d) for j in range(q)]


def frob(M):
    return np.linalg.norm(M, "fro")


def setup(d, p, q, alpha):
    E = prior_frame(d, p)
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE, _ = np.linalg.qr(GE)
    A_, B_, F_ = build_F_at_angle(QE, q, alpha, d)
    Lam = np.array([[np.trace(E[i] @ E[j].T) for j in range(p)] for i in range(p)])
    H = np.array([[np.trace(E[i] @ A_[j].T) for j in range(q)] for i in range(p)])
    Gop = np.zeros((p + q, p + q))
    Gop[:p, :p] = Lam
    Gop[:p, p:] = np.cos(alpha) * H
    Gop[p:, :p] = np.cos(alpha) * H.T
    Gop[p:, p:] = np.eye(q)
    C = np.linalg.solve(Lam, H)
    return E, A_, B_, F_, Lam, H, Gop, C


out = []
P, Q = 4, 2

out.append("=== Q0  analytic Gram vs true Frobenius Gram of {E_i} U {F_j};  and H^T Lam^{-1} H = I ? ===")
for d in (8, 16, 32):
    a = np.deg2rad(10.0)
    diff, ident = [], []
    for _ in range(10):
        E, A_, B_, F_, Lam, H, Gop, C = setup(d, P, Q, a)
        M = E + F_
        Gt = np.array([[np.trace(M[i] @ M[j].T) for j in range(P + Q)] for i in range(P + Q)])
        diff.append(np.abs(Gt - Gop).max())
        ident.append(np.abs(H.T @ np.linalg.solve(Lam, H) - np.eye(Q)).max())
    out.append("d=%3d  max|G_true - G_analytic| = %.2e   max|H^T Lam^-1 H - I| = %.2e"
               % (d, np.mean(diff), np.mean(ident)))
out.append("")

out.append("=== Q1  lam_min(Gop) vs sin^2 a/(1+cos^2 a ||C||_2^2);  ||C||_2^2 vs 1/d (=1/rho_E^2) ===")
out.append("%4s %6s %13s %13s %8s %10s %10s" % ("d", "alpha", "lam_min", "formula", "ratio", "||C||^2", "d*||C||^2"))
for d in (4, 8, 16, 32, 64):
    for adegs in (2.0, 10.0, 45.0, 80.0):
        a = np.deg2rad(adegs)
        rat, lm_s, cn_s = [], [], []
        for _ in range(8):
            E, A_, B_, F_, Lam, H, Gop, C = setup(d, P, Q, a)
            lm = np.linalg.eigvalsh(Gop)[0]
            cn2 = np.linalg.norm(C, 2) ** 2
            rat.append(lm / (np.sin(a) ** 2 / (1.0 + np.cos(a) ** 2 * cn2)))
            lm_s.append(lm); cn_s.append(cn2)
        out.append("%4d %6.1f %13.5e %13.5e %8.4f %10.4f %10.4f"
                   % (d, adegs, np.mean(lm_s), np.mean([np.sin(a) ** 2 / (1 + np.cos(a) ** 2 * c) for c in cn_s]),
                      np.mean(rat), np.mean(cn_s), np.mean(cn_s) * d))
out.append("")

out.append("=== Q2  eps_n = sig_min(G)/(sqrt n * sig_min(Gop^1/2)) - 1   (d=10, p+q=6) ===")
out.append("%6s %6s %12s %12s %14s" % ("n", "alpha", "mean|eps|", "max|eps|", "|eps|/sqrt((p+q)/n)"))
d = 10
for n in (100, 200, 400, 800, 1600, 3200):
    for adegs in (1.0, 5.0, 30.0, 90.0):
        a = np.deg2rad(adegs)
        eps = []
        for _ in range(12):
            E, A_, B_, F_, Lam, H, Gop, C = setup(d, P, Q, a)
            lm = max(np.linalg.eigvalsh(Gop)[0], 1e-14)
            U = rng.normal(size=(d, n))
            G = np.column_stack([np.reshape(M @ U, -1) for M in E + F_])
            sG = np.linalg.svd(G, compute_uv=False)[-1]
            eps.append(sG / (np.sqrt(n * lm)) - 1.0)
        eps = np.array(eps)
        out.append("%6d %6.1f %12.5f %12.5f %14.4f"
                   % (n, adegs, np.mean(np.abs(eps)), np.max(np.abs(eps)),
                      np.mean(np.abs(eps)) / np.sqrt((P + Q) / n)))
out.append("")

out.append("=== Q2b  d=24, alpha as small as 0.3 deg: does eps stay alpha-free? (naive bound would explode) ===")
for n in (200, 800, 3200):
    for adegs in (0.3, 1.0, 3.0, 20.0):
        a = np.deg2rad(adegs)
        eps = []
        for _ in range(10):
            E, A_, B_, F_, Lam, H, Gop, C = setup(24, P, Q, a)
            lm = max(np.linalg.eigvalsh(Gop)[0], 1e-16)
            U = rng.normal(size=(24, n))
            G = np.column_stack([np.reshape(M @ U, -1) for M in E + F_])
            eps.append(np.linalg.svd(G, compute_uv=False)[-1] / np.sqrt(n * lm) - 1.0)
        out.append("d=24 n=%5d a=%4.1f  mean|eps|=%.4f  max|eps|=%.4f  naive d/(sqrt n sin^2 a)=%10.1f"
                   % (n, adegs, np.mean(np.abs(eps)), np.max(np.abs(eps)), 24 / (np.sqrt(n) * np.sin(a) ** 2)))
out.append("")

out.append("=== Q3/Q4  mechanism: ratio r(M)=||MM^T||_F/||M||_F^2 at the near-null direction, and implied const ===")
for d in (10, 24):
    for adegs in (1.0, 30.0):
        a = np.deg2rad(adegs)
        rr, ss = [], []
        for _ in range(12):
            E, A_, B_, F_, Lam, H, Gop, C = setup(d, P, Q, a)
            w, V = np.linalg.eigh(Gop)
            v = V[:, 0]
            Mx = sum(v[i] * E[i] for i in range(P)) + sum(v[P + j] * F_[j] for j in range(Q))
            rr.append(frob(Mx @ Mx.T) / frob(Mx) ** 2)
            ss.append(frob(Mx) / frob(np.eye(d)) * np.sqrt(d))   # scale sanity: ~||Mx||_F vs 1
        out.append("d=%3d a=%4.1f  r = %.4f (<=1 always, isotropic ~ 1/sqrt(d)=%.3f)"
                   % (d, adegs, np.mean(rr), 1 / np.sqrt(d)))
out.append("")

out.append("=== Q5  dictionary size: eps vs sqrt((p+q)/n) at n=600, d=10, a=5deg ===")
n, a = 600, np.deg2rad(5.0)
for (p, q) in ((1, 1), (2, 1), (4, 2), (8, 4), (16, 8)):
    eps = []
    for _ in range(10):
        E, A_, B_, F_, Lam, H, Gop, C = setup(d, p, q, a)
        lm = max(np.linalg.eigvalsh(Gop)[0], 1e-14)
        U = rng.normal(size=(d, n))
        G = np.column_stack([np.reshape(M @ U, -1) for M in E + F_])
        eps.append(np.linalg.svd(G, compute_uv=False)[-1] / np.sqrt(n * lm) - 1.0)
    out.append("p+q=%3d  mean|eps|=%.4f  /sqrt((p+q)/n)=%.3f" % (p + q, np.mean(np.abs(eps)),
                                                                  np.mean(np.abs(eps)) / np.sqrt((p + q) / n)))
out.append("")

out.append("=== Q6  pooled worst-case: max over 40 draws of (1+eps)^2 - 1 vs the Wishart relative-deviation scale ===")
for n in (200, 600, 1800):
    a = np.deg2rad(5.0)
    bad = []
    for _ in range(40):
        E, A_, B_, F_, Lam, H, Gop, C = setup(10, P, Q, a)
        lm = max(np.linalg.eigvalsh(Gop)[0], 1e-14)
        U = rng.normal(size=(10, n))
        G = np.column_stack([np.reshape(M @ U, -1) for M in E + F_])
        sG2 = np.linalg.svd(G, compute_uv=False)[-1] ** 2
        bad.append(sG2 / (n * lm) - 1.0)
    bad = np.array(bad)
    out.append("n=%4d  min(1+eps^2)=%.4f  max=%.4f  ->  worst downward rel. dev of lam_min = %.4f  vs sqrt((p+q)/n)=%.4f"
               % (n, 1 + bad.min(), 1 + bad.max(), -bad.min(), np.sqrt(6 / n)))

print("\n".join(out))
