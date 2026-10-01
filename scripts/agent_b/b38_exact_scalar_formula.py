# B round 19g: EXACT formula for lambda_min(Gcal) -- the sphere minimisation collapses.
#
# From b37, w0 is NOT the minimiser of mu(w). But max over the unit sphere of Phi_w has a closed form:
#   Phi_w(l) = l + cos^2 a * l * y^T Lam (Lam - l I)^{-1} y - s,      y = C w,   s = sin^2 a
#   (combine l(1+||u||^2) + l^2 u^T(Lam-lI)^{-1}u with u = cos(a) y, and I + l(Lam-lI)^{-1} = Lam(Lam-lI)^{-1})
#   max_{||w||=1} Phi_w(l) =: Psi(l) = l + cos^2(a) l lambda_max(B(l)) - s,  B(l) = C^T Lam (Lam-lI)^{-1} C
# Psi is strictly increasing on (0, lam_min(Lam)) (Lam(Lam-lI)^{-1} = (I - l Lam^{-1})^{-1} increases in
# Loewner order), Phi_w <= Psi, and equality holds at w* = top eigenvector of B(l*). Therefore
#   min_{||w||=1} mu(w) = the root l* of Psi,   attained at w* = top eigenvector of B(l*),
#   lambda_min(Gcal) = min( lambda_min(Lam), l* )   (the w=0 face of the sphere gives lambda_min(Lam)).
# Hypotheses, written BEFORE running:
#   (E1) |l*_pred/lam - 1| <= 1e-12 on every draw -- this is now an EXACT formula, not a bound.
#   (E2) the min() is exercised: at least one cell with lambda_min(Lam) < l* (ill-conditioned prior Gram),
#        where the formula must return lambda_min(Lam), i.e. the w=0 face wins.
#   (E3) the optimal direction is NOT w0 in general: angle(w*, w0) > 0 measurably often, growing with alpha.
#   (E4) corollary check: l* <= UB = s/(1+cos^2(a) t) follows from B(l) >= C^T C, so the theorem must
#        reproduce Theorem 3.1's upper side rather than contradict it.
import numpy as np

rng = np.random.default_rng(43)
orth = lambda A: np.linalg.qr(A)[0]


def lam_pred(Lam, C, al, s):
    ell, Vec = np.linalg.eigh(Lam)
    lmin = ell[0]
    c2 = np.cos(al) ** 2
    Vc = C.T @ Vec                      # (q,p): C^T in Lam's eigenbasis

    def topB(l):                        # lambda_max(C^T Lam (Lam-lI)^{-1} C)
        w = ell / (ell - l)
        M = (Vc * w) @ Vc.T
        return np.linalg.eigvalsh(M)[-1]

    def Psi(l):
        return l + c2 * l * topB(l) - s

    lo, hi = 1e-16, lmin * (1 - 1e-10)
    if Psi(hi) <= 0:
        return hi, None, lmin            # no interior root: the w=0 face is the min
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Psi(mid) < 0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    w = ell / (ell - r)
    M = (Vc * w) @ Vc.T
    return r, M, lmin


rows = []
DRAW = 120
print("%-9s %4s | %11s %11s %6s | %8s %8s" %
      ("cell", "alpha", "max|pred/lam-1|", "max|l*/UB-1|", "w0-face", "ang(w*,w0)", "viol"))
for d, P, Q in ((3, 2, 2), (4, 2, 2), (5, 3, 3), (6, 3, 3), (8, 4, 2), (8, 4, 3),
                (10, 4, 3), (12, 5, 4), (20, 6, 4)):
    for alpha_deg in (1, 20, 45, 70, 85, 89):
        al = np.deg2rad(alpha_deg)
        s = np.sin(al) ** 2
        e1, e4, face, ang, viol = 0.0, 0.0, 0, [], 0
        for _ in range(DRAW):
            E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
            V = np.column_stack([e.reshape(-1) for e in E])
            Lam = V.T @ V
            QE, _ = np.linalg.qr(V)
            a = orth(QE @ orth(rng.normal(size=(P, Q))))
            br = rng.normal(size=(d * d, Q))
            b = orth(br - QE @ (QE.T @ br))
            FB = np.cos(al) * a + np.sin(al) * b
            H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)]
                          for i in range(P)])
            C = np.linalg.solve(Lam, H)
            lam = np.linalg.eigvalsh(np.block([[Lam, np.cos(al) * H],
                                               [np.cos(al) * H.T, np.eye(Q)]]))[0]
            t = np.linalg.norm(C, 2) ** 2
            UB = s / (1 + np.cos(al) ** 2 * t)
            r, M, lmin = lam_pred(Lam, C, al, s)
            pred = min(lmin, r)
            e1 = max(e1, abs(pred / lam - 1))
            e4 = max(e4, abs(r / UB - 1))
            face += lmin <= r
            viol += pred > lam * (1 + 1e-9)
            if M is not None:
                wstar = np.linalg.eigh(M)[1][:, -1]
                w0 = np.linalg.svd(C)[2][0]
                ang.append(np.arccos(min(1.0, abs(wstar @ w0))))
        rows.append((e1, e4, face, np.mean(ang) if ang else 0,
                     max(ang) if ang else 0, viol))
        print("%-9s %4d | %11.2e %11.2e %6d | %8.4f %8.4f | %d" %
              ("%dx%dx%d" % (d, P, Q), alpha_deg, e1, e4, face,
                   np.mean(ang) if ang else 0.0, max(ang) if ang else 0.0, viol))
f = np.array([r[:5] for r in rows])
print("\nE1 worst |pred/lam-1| over %d cells x %d draws: %.2e   (1e-12 = round-off, so the formula is exact)"
      % (len(rows), DRAW, max(r[0] for r in rows)))
print("E2 cells where the w=0 face wins (lambda_min(Lam) <= l*): %d cells, %d draws"
      % (int((f[:, 2] > 0).sum()), int(f[:, 2].sum())))
print("E3 mean/max angle between the optimal w* and w0 (radians): %.4f / %.4f" % (f[:, 3].mean(), f[:, 4].max()))
print("E4 worst |l*/UB-1|: %.2e  (l* <= UB must hold: B(l) >= C^T C)" % max(r[1] for r in rows))
print("draws where pred exceeded the exact lambda_min (must be 0): %d" % int(sum(r[5] for r in rows)))
