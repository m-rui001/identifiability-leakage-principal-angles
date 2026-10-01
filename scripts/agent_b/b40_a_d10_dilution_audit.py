# B round 20: audit of A's d10_misspec_angle (the "err ~ c*(noise+model error)/sin(alpha)" law).
#
# Read from A's script, not from his summary. Two things looked wrong before running:
#   (K1) his "predicted c*||D||/sin(a)" column is computed as row[-1]*sin(a)/0.1 from the SAME measurement
#        it is displayed next to -> it cannot disagree, so it is not a check of anything.
#   (K2) his fitted constant c ~ 0.14-0.19 has a candidate closed form: A's Delta_hidden is a random
#        Gaussian operator of fixed Frobenius norm, so its component inside the q-dimensional correction
#        span is ||P_F Delta|| ~ ||Delta||*sqrt(q)/d^2^{1/2} = ||Delta||*sqrt(q)/d. At q=3, d=10 that is
#        0.1732. If true, c is a DIMENSIONAL DILUTION FACTOR (chapter Cor. 5.2), not a physical constant,
#        and it must be independent of n while the noise constant must go like sqrt(q/n) (A's G3 measures
#        0.064, and sqrt(3/600)=0.0707). One constant cannot cover both.
# FALSIFIABLE FORMS (all checked, printed with signed deviations):
#   K1  pred_alpha == err(||D||=0.1)*sin(alpha)/0.1 to round-off on A's own numbers.
#   K2  c1(d) := err*sin(alpha)/||D|| satisfies c1*sqrt(q)^{-1}*d ~ 1, i.e. c1 ~ sqrt(q)/d, over d=6..20.
#   K3  c1 is FLAT in n (n=150..2400) while c2 := err*sin(alpha)/sigma satisfies c2*sqrt(n/q) ~ 1.
#   K4  c2 is FLAT in d (the noise arm does not know about d), c1 does not know about n.
# A's construction is reproduced verbatim (seed 60601, same helper, same lstsq) so that a disagreement
# can only be about the interpretation, not about the setup.
import numpy as np

rng = np.random.default_rng(60601)


def orth(A):
    return np.linalg.qr(A)[0]


def build_F_at_angle(QE, p, q, alpha):
    QE = orth(QE[:, :p])
    a = orth(QE @ orth(rng.normal(size=(p, q))))
    b_raw = rng.normal(size=(d * d, q))
    b = orth(b_raw - QE @ (QE.T @ b_raw))
    return np.cos(alpha) * a + np.sin(alpha) * b


d = 10


def run(p, q, alpha_deg, delta_norm, noise, n_samples=600, trials=30):
    alpha = np.deg2rad(alpha_deg)
    errs = []
    for _ in range(trials):
        E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(p)]
        GE = np.column_stack([e.reshape(-1) for e in E])
        QE, _ = np.linalg.qr(GE)
        FB = build_F_at_angle(QE, p, q, alpha)
        F = [FB[:, j].reshape(d, d) for j in range(q)]
        theta = rng.normal(size=p); psi = rng.normal(size=q)
        Dh = rng.normal(size=(d, d)); Dh = Dh / np.linalg.norm(Dh, "fro") * delta_norm
        A = (sum(theta[i] * E[i] for i in range(p)) + sum(psi[j] * F[j] for j in range(q)) + Dh)
        U = rng.normal(size=(d, n_samples))
        Y = A @ U + rng.normal(scale=noise, size=(d, n_samples))
        G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
        sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
        C_hat = sum(sol[p + j] * F[j] for j in range(q))
        C = sum(psi[j] * F[j] for j in range(q))
        errs.append(np.linalg.norm(C_hat - C, "fro"))
    return np.mean(errs)


out = []

# ---------- K0: reproduce A's [G1] rows exactly, then K1: show the pred column is an identity --------
out.append("=== K0/K1  A's d10 [G1] reproduced (seed 60601) and his 'prediction' column inspected ===")
out.append("%5s %11s %11s %11s %11s %12s" % ("alpha", "err .01", "err .05", "err .1", "A's pred", "pred/row*sin/.1"))
for a in [2, 5, 10]:
    row = [run(4, 3, a, dn, 1e-4, trials=15) for dn in [0.01, 0.05, 0.1]]
    pred = row[-1] / (0.1 / np.sin(np.deg2rad(a)))
    ident = pred * 0.1 / np.sin(np.deg2rad(a))
    out.append("%5d %11.4e %11.4e %11.4e %11.4f %12.3e" % (a, row[0], row[1], row[2], pred, ident / row[2]))
out.append("last column == 1 means A's 'predicted c*||D||/sin(a)' IS err(||D||=0.1)*sin(a)/0.1, i.e. the")
out.append("fitted constant of that very row: the column has no independent content (K1).")
out.append("")

# ---------- K2: the model-form constant against sqrt(q)/d over dictionary size ----------------------
out.append("=== K2  model-form arm: c1 := err*sin(alpha)/||D||  vs  sqrt(q)/d   (q=3, alpha=20 deg, n=600) ===")
p, q, a_deg, dn = 4, 3, 20.0, 0.05
out.append("%4s %11s %11s %10s %10s" % ("d", "c1", "sqrt(q)/d", "c1*d/sqrtq", "dev"))
for dd in (6, 8, 10, 14, 20):
    global_d = dd
    # same construction at a different d: rebind the module-level d used by the helpers
    globals()["d"] = dd
    c1 = run(p, q, a_deg, dn, 1e-4, trials=100) * np.sin(np.deg2rad(a_deg)) / dn
    ref = np.sqrt(q) / dd
    out.append("%4d %11.4e %11.4e %10.3f %10.2e" % (dd, c1, ref, c1 / ref, c1 / ref - 1.0))
globals()["d"] = 10
out.append("K2 is satisfied if c1*d/sqrt(q) stays near 1: then A's c is the dilution factor, not physics.")
out.append("")

# ---------- K3/K4: n-scaling of the two arms, and d-blindness of the noise arm ----------------------
out.append("=== K3  n-scaling: model-form arm (c1) vs noise arm (c2 := err*sin/sigma), d=10, q=3 ===")
out.append("%6s %11s %11s %12s %12s" % ("n", "c1", "c2", "c2*sqrt(n/q)", "c2/c1"))
for n in (150, 300, 600, 1200, 2400):
    c1 = run(p, q, a_deg, dn, 1e-4, n_samples=n, trials=60) * np.sin(np.deg2rad(a_deg)) / dn
    c2 = run(p, q, a_deg, 0.0, 1e-2, n_samples=n, trials=60) * np.sin(np.deg2rad(a_deg)) / 1e-2
    out.append("%6d %11.4e %11.4e %12.3f %12.3f" % (n, c1, c2, c2 * np.sqrt(n / q), c2 / c1))
out.append("")
out.append("=== K4  d-scaling of the NOISE arm at fixed n=600 (c2 must not know about d) ===")
out.append("%4s %11s %11s" % ("d", "c2", "sqrt(q/n)"))
for dd in (6, 10, 14, 20):
    globals()["d"] = dd
    c2 = run(p, q, a_deg, 0.0, 1e-2, n_samples=600, trials=60) * np.sin(np.deg2rad(a_deg)) / 1e-2
    out.append("%4d %11.4e %11.4e" % (dd, c2, np.sqrt(q / 600.0)))
globals()["d"] = 10
out.append("")
out.append("Verdict wording for the chapter: err ~ [sqrt(q)/d]*||D||/sin a + [sqrt(q/n)]*sigma/sin a.")
out.append("A's single fitted c averages the two arms (his own G2/G3 print 0.16 and 0.064), which is the")
out.append("two-constants problem the Introduction objects to; his alpha-scaling 1/sin a is NOT in doubt.")
print("\n".join(out))
