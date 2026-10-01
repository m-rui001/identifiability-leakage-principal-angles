# B round 19f: is w0 (top right-singular direction of C) really the minimiser of mu(w) over the sphere?
#
# Remark 3.5 of the chapter states the EXACT reduction
#   lambda_min(Gcal) = min( lam_min(Lam),  min_{||w||=1} mu(w) ),
#   mu(w) = the root of  Phi_w(l) = l(1+||u||^2) + l^2 u^T (Lam - l I)^{-1} u - s,  u = cos(a) C w,
# and reports that a RANDOM unit w never beat w0 in 3200 draws. Random points on S^{q-1} are weak
# evidence: the competing directions are near-degenerate in sigma_2(C). This script OPTIMISES over the
# sphere (dense grid for q=2, grid+refine for q=3) and tests three things separately:
#   (V1) the exact reduction itself:  min_sphere mu  ==  lambda_min(Gcal)  (this is a PROVED identity,
#        so any disagreement is a bug in my root solver, not new mathematics -- run it as a check)
#   (V2) w0-optimality:  does mu(w0) == min_sphere mu, or can a better w be found?
#   (V3) if V2 fails, how much does lambda_min move? (the practical question for the chapter)
# mu is evaluated with Lam's eigendecomposition, and the bisection is vectorised over the whole grid.
import numpy as np

rng = np.random.default_rng(31)
orth = lambda A: np.linalg.qr(A)[0]


def mu_vec(ell, Ucomp, s, lmin, iters=80):
    """roots of Phi on (0, lmin) for a stack of u-vectors given in Lam's eigenbasis:
       Phi(l) = l(1+nun) + l^2 * sum_i comp_i^2/(ell_i - l) - s, increasing in l."""
    nun = np.einsum("gi,gi->g", Ucomp, Ucomp)
    lo = np.full(nun.shape, 1e-16)
    hi = np.full(nun.shape, lmin * (1 - 1e-9))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = mid * (1.0 + nun) + mid ** 2 * np.sum(
            Ucomp ** 2 / (ell[None, :] - mid[:, None]), axis=1) - s
        neg = res < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def sphere(Q, G):
    if Q == 2:
        th = np.linspace(0, 2 * np.pi, G, endpoint=False)
        return np.column_stack([np.cos(th), np.sin(th)])
    W = rng.normal(size=(G, Q))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


rows = []
DRAW, GRID = 30, 4000
print("cell          alpha  lam/mu0-1      min mu vs lam   w0 fails   best-w gain   grid-vs-exact")
for d, P, Q in ((8, 4, 2), (8, 4, 3), (10, 4, 2), (10, 4, 3), (12, 5, 3), (6, 3, 3), (5, 2, 2)):
    for alpha_deg in (2, 20, 45, 85):
        al = np.deg2rad(alpha_deg)
        s = np.sin(al) ** 2
        ell_ref, dev, fails, gains, v1 = 0.0, [], 0, [], 0.0
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
            Gop = np.block([[Lam, np.cos(al) * H], [np.cos(al) * H.T, np.eye(Q)]])
            lam = np.linalg.eigvalsh(Gop)[0]
            ell, Vec = np.linalg.eigh(Lam)
            lmin = ell[0]
            Wg = sphere(Q, GRID)
            Ucomp = np.cos(al) * (Wg @ C.T) @ Vec          # grid of u-vectors in Lam's eigenbasis
            mu_g = mu_vec(ell, Ucomp, s, lmin)
            w0 = np.linalg.svd(C)[2][0]
            mu0 = mu_vec(ell, (np.cos(al) * (w0 @ C.T)) @ Vec[None, :], s, lmin)[0]
            mubest = mu_g.min()
            dev.append(lam / mu0 - 1)
            gains.append(mu0 / mubest - 1)
            fails += mubest < mu0 - 1e-10 * max(mu0, 1e-12)
            v1 = max(v1, abs(mubest / lam - 1))
            ell_ref = max(ell_ref, lam / lmin)
        dv = np.array(dev)
        print("%-12s %4d  %+.3e      %+.3e        %3d/%d      %+.3e      %.2e" %
              ("%dx%dx%d" % (d, P, Q), alpha_deg, np.mean(dv), max(abs(dv)),
               fails, DRAW, np.mean(gains), v1))
        rows.append((fails, v1, np.mean(gains)))
f = np.array(rows)
print("\nw0-optimality violations (draws where a sphere point beats w0): %d / %d draws" %
      (f[:, 0].sum(), len(rows) * DRAW))
print("max |min_sphere_mu / lambda_min - 1| over all cells: %.2e  "
      "(this column IS the proved identity check: it must be grid-coarseness, not a sign flip)" % f[:, 1].max())
print("mean relative gain of the best grid w over w0: %.3e" % f[:, 2].mean())
