# B round 19e: the CHAIN  lambda_min(Gcal) <= mu(w0) <= cf(t,alpha) <= UB.
#
# Why cf is no longer a guess: cf is the Rayleigh minimum restricted to a PLANE.
#   quotient Q(x,w) = (x^T Lam x + s||w||^2)/(||x - cos(a) C w||^2 + ||w||^2),  s = sin^2 a, t = ||C||_2^2.
#   Take w = w0 (unit top right-singular vector of C) and x = c*u, u = cos(a) C w0.
#   The isometry C^T Lam C = I_q gives  u^T Lam u = cos^2(a),  ||u||^2 = cos^2(a) * t,
#   so the 2x2 pencil A = diag(cos^2 a, s), B = [[p,-p],[-p,p+1]] with p = cos^2(a) t has entries in
#   (a, t) ONLY. det(A - lam B) = 0 reduces to  t lam^2 - (1+t) lam + s = 0,  whose SMALLER root is cf.
# Each step of the chain is a restriction of the previous feasible set, so the whole chain is provable:
#   (I)  lam_min <= mu(w0)     (fix w)
#   (II) mu(w0) <= cf          (further restrict x to span{u})
#   (III) cf <= UB             (algebra: Q(cf-branch) - see text)
#   (IV) cf = min over the plane (numeric check against brute-force 1-D minimisation)
import numpy as np
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(23)
orth = lambda A: np.linalg.qr(A)[0]


def make(d, P, Q, al):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
    V = np.column_stack([e.reshape(-1) for e in E])
    Lam = V.T @ V
    QE, _ = np.linalg.qr(V)
    a = orth(QE @ orth(rng.normal(size=(P, Q))))
    br = rng.normal(size=(d * d, Q))
    b = orth(br - QE @ (QE.T @ br))
    FB = np.cos(al) * a + np.sin(al) * b
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)] for i in range(P)])
    return Lam, np.linalg.solve(Lam, H), np.block([[Lam, np.cos(al) * H],
                                                   [np.cos(al) * H.T, np.eye(Q)]])


def mu_of(Lam, u, s, lmin):
    nun = float(u @ u)
    I = np.eye(Lam.shape[0])

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


rows = []
DRAW = 150
print("%3s %3s %3s %4s %6s %8s | %8s %9s | %s" %
      ("d", "p", "q", "alpha", "min t", "max t", "max cf/UB", "max|pl-cf|",
       "viol I  II  III  IV"))
for d, P, Q in ((3, 2, 2), (4, 2, 2), (6, 3, 3), (8, 4, 3), (10, 4, 3), (12, 5, 4), (16, 4, 3)):
    for alpha_deg in (1, 30, 60, 75, 85, 89):
        al = np.deg2rad(alpha_deg)
        s = np.sin(al) ** 2
        c2 = np.cos(al) ** 2
        vI = vII = vIII = vIV = 0
        tmin, tmax, rmax, dmax = np.inf, 0.0, 0.0, 0.0
        for _ in range(DRAW):
            Lam, C, Gop = make(d, P, Q, al)
            t = np.linalg.norm(C, 2) ** 2
            tmin, tmax = min(tmin, t), max(tmax, t)
            lam = np.linalg.eigvalsh(Gop)[0]
            UB = s / (1 + c2 * t)
            cf = ((1 + t) - np.sqrt(max((1 + t) ** 2 - 4 * t * s, 0.0))) / (2 * t)
            _, _, V2 = np.linalg.svd(C)
            u = np.cos(al) * (C @ V2[0])
            mu = mu_of(Lam, u, s, np.linalg.eigvalsh(Lam)[0])
            # (IV) brute force over the plane: min_c (c^2 cos^2a + s)/(||u||^2 (c-1)^2 + 1)
            nun = float(u @ u)
            f = lambda c: (c * c * c2 + s) / (nun * (c - 1) ** 2 + 1.0)
            plane = minimize_scalar(f, bounds=(-1e4, 1e4), method="bounded",
                                    options={"xatol": 1e-13}).fun
            vI += lam > mu * (1 + 1e-9)
            vII += mu > cf * (1 + 1e-9)
            vIII += cf > UB * (1 + 1e-9)
            vIV += abs(plane / cf - 1) > 1e-7
            rmax = max(rmax, cf / UB)
            dmax = max(dmax, abs(plane / cf - 1))
        print("%3d %3d %3d %4d %6.3f %8.3f | %8.5f %9.1e | %4d %3d %3d %3d" %
              (d, P, Q, alpha_deg, tmin, tmax, rmax, dmax, vI, vII, vIII, vIV))
        rows.append((vI, vII, vIII, vIV))
tot = np.array(rows).sum(0)
print("\ntotals over %d cells x %d draws: I=%d II=%d III=%d IV=%d (all must be 0)" %
      (len(rows), DRAW, *tot))
print("cf/UB < 1 everywhere means the plane bound is strictly sharper than Theorem 3.1's branch.")
