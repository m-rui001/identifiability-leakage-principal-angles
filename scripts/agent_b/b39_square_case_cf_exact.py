# B round 19h: corollary of the exact formula -- when q = p, lambda_min(Gcal) IS cf(t,alpha) EXACTLY.
#
# If C (p x q) is square and invertible, the isometry C^T Lam C = I_q forces Lam = C^{-T} C^{-1}, hence
#   B(l) = C^T Lam (Lam - lI)^{-1} C = (I - l C^T C)^{-1} C^T C =: f(S),  S = C^T C, f(x) = x/(1-l x),
# a FUNCTION OF S alone. f is increasing, so lambda_max(B(l)) = f(sigma_1^2) = t/(1-l t) is attained at w0.
# The root equation  l + cos^2(a) l t/(1-l t) = s  times (1-lt) is exactly  t l^2 - (1+t) l + s = 0.
#   => q = p  :  lambda_min(Gcal) = cf(t,alpha), closed form, no eigenvalue computation.
#   => q < p  :  w0 is NOT optimal in general (b37), the exact value needs lambda_max(B(l)).
# (E1) square cells: |lam/cf - 1| <= 1e-12; (E2) rectangular cells: lam < cf strictly, by a real gap;
# (E3) the identity Lam = C^{-T}C^{-1} holds to round-off on square cells (that is what makes it work).
import numpy as np

rng = np.random.default_rng(57)
orth = lambda A: np.linalg.qr(A)[0]
DRAW = 200
print("%-9s %4s | %12s %12s %10s | %10s %10s" %
      ("cell", "alpha", "max|lam/cf-1|", "mean lam/cf", "min lam/cf", "|C| rank", "Lam-resid"))
for d, P, Q in ((3, 2, 2), (4, 2, 2), (5, 3, 3), (6, 3, 3), (8, 4, 4), (10, 3, 3), (12, 5, 5),
                (8, 4, 3), (10, 4, 3), (12, 5, 4), (6, 4, 3)):
    for alpha_deg in (1, 20, 45, 70, 85):
        al = np.deg2rad(alpha_deg)
        s = np.sin(al) ** 2
        dev, rat, res, sq = [], [], [], P == Q
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
            t = np.linalg.norm(C, 2) ** 2
            cf = ((1 + t) - np.sqrt(max((1 + t) ** 2 - 4 * t * s, 0.0))) / (2 * t)
            lam = np.linalg.eigvalsh(np.block([[Lam, np.cos(al) * H],
                                               [np.cos(al) * H.T, np.eye(Q)]]))[0]
            dev.append(abs(lam / cf - 1))
            rat.append(lam / cf)
            if sq:
                res.append(np.linalg.norm(Lam - np.linalg.inv(C).T @ np.linalg.inv(C), 2)
                           / np.linalg.norm(Lam, 2))
        print("%-9s %4d | %12.2e %12.6f %10.6f | %10s %10s" %
              ("%dx%dx%d" % (d, P, Q), alpha_deg, max(dev), np.mean(rat), min(rat),
               "p=q" if sq else "p>q", ("%.2e" % max(res)) if sq else "--"))
print("\nsquare cells must give |lam/cf-1| at round-off (the formula IS cf); rectangular cells must give")
print("lam/cf strictly below 1 (cf is then only an upper bound, and the gap is the q<p defect).")
