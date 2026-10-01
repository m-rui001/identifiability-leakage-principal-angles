# B round 19c: REDUCTION of lambda_min(Gcal) to a scalar equation.
#
# From the Rayleigh form (chapter eq:rayleigh), with x = Lambda_E^{-1/2} z (bijective, Lambda_E > 0) and
# C = Lambda_E^{-1} H, t = ||C||_2^2, s = sin^2 alpha:
#   lambda_min(Gcal) = min_{(x,w)} (x^T Lam x + s ||w||^2) / ( ||x - cos(alpha) C w||^2 + ||w||^2 ).
# FIX w = w0 (a unit top right-singular vector of C) and minimize over x only. That restricted minimum
#   mu := min_x (x^T Lam x + s)/(||x - u||^2 + 1),   u = cos(alpha) C w0,
# is an UPPER bound on lambda_min(Gcal) by construction, and by the S-lemma / fractional programming it is
# the largest lambda with (x^T Lam x + s) - lambda(||x-u||^2+1) >= 0 for all x, i.e. the root of
#   Phi(lambda) := lambda (1 + ||u||^2) + lambda^2 u^T (Lam - lambda I)^{-1} u - s = 0,  0 < lambda < lam_min(Lam).
# HYPOTHESES, written before running (falsifiable forms):
#   (H1) lambda_min(Gcal) <= mu on every draw  -- this is the provable direction, it must never fail.
#   (H2) the bound is essentially EXACT: |lambda_min/mu - 1| << |lambda_min/UB - 1| (UB = s/(1+cos^2 t)).
#   (H3) w0 is the right choice: mu(w0) <= mu(w) for random unit w, and mu(w0) < mu(w_bottom).
#   (H4) cf(t,alpha), which uses only t, approximates mu: |mu/cf - 1| small but NOT machine-zero (cf is a
#        approximation of mu, not mu itself).
import numpy as np

rng = np.random.default_rng(7)
orth = lambda A: np.linalg.qr(A)[0]


def build(QE, p, q, alpha, d):
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    br = rng.normal(size=(d * d, q))
    b = orth(br - QE @ (QE.T @ br))
    return a, np.cos(alpha) * a + np.sin(alpha) * b


def mu_of(Lam, u, s, lo_hi):
    """root of Phi on (0, lam_min(Lam)); Phi is increasing there."""
    lmin = lo_hi
    P = Lam.shape[0]
    I = np.eye(P)
    nun = float(u @ u)

    def Phi(l):
        return l * (1.0 + nun) + l * l * float(u @ np.linalg.solve(Lam - l * I, u)) - s

    lo, hi = 1e-16, lmin * (1 - 1e-9)
    if Phi(hi) < 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


out = []
DRAW = 200
out.append("=== restricted-minimum mu vs exact lambda_min(Gcal), %d draws per cell ===" % DRAW)
out.append("%3s %3s %3s %3s %9s %11s %11s %11s %11s %11s %6s" %
           ("d", "p", "q", "alpha", "lam/UB-1", "lam/mu-1", "max|lam/mu-1|",
            "mu/cf-1", "max|mu/cf-1|", "H1 fails", "w>u"))
for d, P, Q in ((8, 4, 3), (10, 4, 3), (12, 5, 4), (6, 3, 3)):
    for alpha_deg in (2, 20, 45, 85):
        al = np.deg2rad(alpha_deg)
        s = np.sin(al) ** 2
        devU, devM, muU, muC, devC, h1, worse = [], [], [], [], [], 0, 0
        for _ in range(DRAW):
            E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
            V = np.column_stack([e.reshape(-1) for e in E])
            Lam = V.T @ V
            QE, _ = np.linalg.qr(V)
            a, FB = build(QE, P, Q, al, d)
            H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)]
                         for i in range(P)])
            C = np.linalg.solve(Lam, H)
            t = np.linalg.norm(C, 2) ** 2
            Gop = np.block([[Lam, np.cos(al) * H], [np.cos(al) * H.T, np.eye(Q)]])
            lam = np.linalg.eigvalsh(Gop)[0]
            UB = s / (1 + np.cos(al) ** 2 * t)
            cf = ((1 + t) - np.sqrt(max((1 + t) ** 2 - 4 * t * s, 0.0))) / (2 * t)
            _, sv2, V2 = np.linalg.svd(C)
            u = np.cos(al) * (C @ V2[0])
            mu = mu_of(Lam, u, s, np.linalg.eigvalsh(Lam)[0])
            devU.append(abs(lam / UB - 1)); devM.append(lam / mu - 1)
            muU.append(mu / UB - 1); muC.append(cf); devC.append(abs(mu / cf - 1))
            h1 += lam > mu * (1 + 1e-9)
            # control: is w0 the best correction-block direction?
            wr = rng.normal(size=Q); wr /= np.linalg.norm(wr)
            mur = mu_of(Lam, np.cos(al) * (C @ wr), s, np.linalg.eigvalsh(Lam)[0])
            worse += mur < mu * (1 - 1e-9)
        dm = np.array(devM)
        out.append("%3d %3d %3d %3d %9.2e %11.2e %11.2e %11.2e %11.2e %6d %6d/%d" %
                   (d, P, Q, alpha_deg, max(devU), abs(np.mean(dm)), max(abs(dm)),
                    np.mean(devC), max(devC), h1, worse, DRAW))
out.append("")
out.append("columns: lam/UB-1 = worst relative slack of the chapter's upper branch (Theorem 3.1);")
out.append("lam/mu-1 = the SAME for the restricted minimisation mu. H1 fails = draws where lam>mu (must be 0).")
out.append("w>u = draws where a RANDOM unit w gave a SMALLER restricted minimum than w0 (H3 violation rate).")
print("\n".join(out))
