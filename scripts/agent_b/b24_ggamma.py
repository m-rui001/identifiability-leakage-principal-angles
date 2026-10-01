# B round 13g: G-gamma, the last open theory item (community.md 23.6 / 25.9).
# The certificate divides by sigma_min(G)^2 = n * lambda_min(Gop), where Gop is the operator
# Frobenius Gram; the quantity the fit actually uses is G^T G / n, whose lambda_min is biased
# DOWNWARD by a one-sided second-order term (b15: gap ~ 1/n, predicted by Rayleigh-Schrodinger).
# G-gamma asks how that bias behaves as alpha -> 0.  lambda_min(Gop) ~ sin^2(a)/(1+cos^2(a)|C|^2),
# so two effects compete: the eigenvalue shrinks AND the gaps in the second-order denominator shrink.
# If bias(alpha) >> sin^2(a) the RELATIVE error of the certificate blows up near alpha = 0 and its
# validity domain must be restricted to sin^2(a) >~ c/n.
# Grams are built numerically from the dictionary columns (no analytic cos alpha factors), so the
# 22.x off-diagonal bug cannot recur.
import numpy as np

RNG = np.random.default_rng(1234)
d, p, q = 8, 4, 3


def draw(alpha):
    """A's build_F_at_angle, verbatim structure: F = cos(a) a_j + sin(a) b_j, a in span(E), b perp span(E)."""
    E = [RNG.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
    GE = np.column_stack([e.reshape(-1) for e in E])          # d^2 x p
    QE = np.linalg.qr(GE)[0]
    a = np.linalg.qr(QE @ np.linalg.qr(RNG.normal(size=(p, q)))[0])[0]     # d^2 x q, orthonormal, in span(E)
    b_raw = RNG.normal(size=(d * d, q))
    b = np.linalg.qr(b_raw - QE @ (QE.T @ b_raw))[0]          # orthonormal, perp span(E)
    GF = np.cos(alpha) * a + np.sin(alpha) * b
    return E, F_list(GF), GE, GF, a, b


def F_list(GF):
    return [GF[:, j].reshape(d, d) for j in range(GF.shape[1])]


rows = []
for n in (200, 600, 2400):
    for adeg in (0.25, 0.5, 1, 2, 5, 10, 20, 45, 70, 85):
        alpha = np.deg2rad(adeg)
        rel, lam_ths, c2s, ratios = [], [], [], []
        for _ in range(300):
            E, F, GE, GF, a, b = draw(alpha)
            big = np.column_stack([GE, GF])
            Gop = big.T @ big
            Lam = GE.T @ GE
            # C = Lambda_E^{-1/2} (GE^T a)  -- the frame invariant of theorem 4.1
            Hh = GE.T @ a
            sv = np.linalg.svd(np.linalg.inv(np.linalg.cholesky(Lam).T) @ Hh, compute_uv=False)
            c2s.append(sv[0] ** 2)
            lt = np.linalg.eigvalsh(Gop)[0]
            lam_ths.append(lt)
            U = RNG.normal(size=(d, n))
            cols = [(Mi @ U).reshape(-1) for Mi in E + F]
            G = np.column_stack(cols)
            le = np.linalg.eigvalsh(G.T @ G / n)[0]
            rel.append(le / lt)
        rel = np.array(rel); lt = np.mean(lam_ths)
        rb = rel.mean() - 1.0
        rows.append((n, adeg, lt, rb, rb / max(np.sin(alpha) ** 2, 1e-12), np.mean(c2s),
                     np.quantile(rel, .005) - 1.0, rel.std()))

print("=== E[lambda_min(G^TG/n)] / lambda_min(Gop) - 1, d=%d p=%d q=%d, 300 draws/cell ===" % (d, p, q))
print("%6s %9s %14s %11s %13s %10s %13s %9s" %
      ("n", "alpha_deg", "lam_min(Gop)", "rel_bias", "rel/sin^2a", "mean|C|^2", "0.5% rel_bias", "sd"))
for r in rows:
    print("%-6d %-9s %-14.4e %-11.5f %-13.4f %-10.4f %-13.5f %-9.4f" % r)
print("")
print("### decision rule:  rel/sin^2a FLAT as alpha->0  => bias ~ sin^2 a => G-gamma TRUE (relative error alpha-independent)")
print("###                 rel/sin^2a RISING             => certificate needs sin^2 a >~ c/n  (state validity domain)")
