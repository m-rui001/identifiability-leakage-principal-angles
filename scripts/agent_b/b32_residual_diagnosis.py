# B round 18d: the "residual" column of b28's table (err/bnd vs q4*q1*q2) is -0.046 at alpha=2 deg
# and -0.0005 at 85 deg, and the chapter never explains it. If eq:factor is an identity, per-trial
# eb must equal per-trial product; so the gap is either (a) a mean-vs-mean artifact (correlation),
# or (b) the idealization err = (dn/cost)/sigma_min(G) is false per trial. This separates them.
import numpy as np

rng = np.random.default_rng(11)
d, P, Q = 10, 4, 3
sigma, dn = 1e-4, 0.05
orth = lambda A: np.linalg.qr(A)[0]


def Cnorm2(E, a):
    Lam = np.column_stack([e.reshape(-1) for e in E])
    Lam = Lam.T @ Lam
    H = np.array([[np.trace(E[i] @ a[:, j].reshape(d, d).T) for j in range(Q)] for i in range(P)])
    return np.linalg.norm(np.linalg.solve(Lam, H), 2) ** 2


def trial(alpha, n):
    E = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(P)]
    GE = np.column_stack([e.reshape(-1) for e in E])
    QE = orth(GE)
    a = orth(QE @ orth(rng.normal(size=(P, Q))))
    br = rng.normal(size=(d * d, Q))
    b = orth(br - QE @ (QE.T @ br))
    FB = np.cos(alpha) * a + np.sin(alpha) * b
    F = [FB[:, j].reshape(d, d) for j in range(Q)]
    lam_G = np.linalg.eigvalsh(np.column_stack([GE, FB]).T @ np.column_stack([GE, FB]))[0]
    psi = rng.normal(size=Q)
    U = rng.normal(size=(d, n))
    G = np.column_stack([(Mi @ U).reshape(-1) for Mi in E + F])
    lam_e = np.linalg.eigvalsh(G.T @ G / n)[0]
    Ug, sv, Vt = np.linalg.svd(G, full_matrices=False)
    W = Ug[:, -1].reshape(d, n)
    Dm = W @ U.T @ np.linalg.inv(U @ U.T)
    cost = np.linalg.norm(Dm, "fro")
    Dm = Dm / cost * dn
    A = sum(rng.normal() * E[i] for i in range(P)) + sum(psi[j] * F[j] for j in range(Q)) + Dm
    Y = A @ U + rng.normal(scale=sigma, size=(d, n))
    sol, *_ = np.linalg.lstsq(G, Y.reshape(-1), rcond=None)
    C_hat = sum(sol[P + j] * F[j] for j in range(Q))
    C = sum(psi[j] * F[j] for j in range(Q))
    e = np.linalg.norm(C_hat - C, "fro")
    LB = np.sin(alpha) ** 2 / (1 + 2 * np.cos(alpha) ** 2 * Cnorm2(E, a))
    eb = e / (dn / np.sqrt(LB))
    q4 = np.sqrt(LB / lam_G)
    q1 = np.sqrt(lam_G / lam_e)
    q2 = 1.0 / (cost * np.sqrt(n))
    ideal = (dn / cost) / (np.sqrt(n * lam_e))          # (dn/cost)/sigma_min(G)
    # the F-block geometry: norm of sum_j v_j F_j for v = last right singular vector
    v = Vt[-1][P:]
    FBnorm = np.linalg.norm(FB @ v, 2)                  # ||sum v_j F_j||_F, columns are vec(F_j)
    return dict(eb=eb, pr=q4 * q1 * q2, e=e, ideal=ideal, q4=q4, q1=q1, q2=q2,
               FBnorm=FBnorm, vn=np.linalg.norm(v), ratio_e=e / ideal,
               q3dict=np.sqrt(lam_G / np.sin(alpha) ** 2))


out = ["per-trial comparison, 60 trials each (means, then the per-trial discrepancy)"]
for alpha_deg, n in ((2, 200), (2, 1800), (20, 200), (85, 200)):
    al = np.deg2rad(alpha_deg)
    rs = [trial(al, n) for _ in range(60)]
    m = lambda k: np.mean([r[k] for r in rs])
    dper = np.array([r["eb"] - r["pr"] for r in rs])
    dideal = np.array([r["ratio_e"] for r in rs])
    # does the block mass q3 = ||v_F|| agree with the dictionary-only surrogate sqrt(lam_G/sin^2)?
    q3 = np.array([r["FBnorm"] for r in rs])
    q3d = np.array([r["q3dict"] for r in rs])
    out.append("alpha=%-3d n=%-5d | mean eb %.4f  mean pr %.4f  mean(eb-pr) %+.5f (sd %.5f)"
               " | e/ideal %.5f+- %.5f | ||Fv|| %.4f ||v|| %.4f"
               " | q3/surrogate mean %+.4f%% max %.4f%%"
               % (alpha_deg, n, m("eb"), m("pr"), dper.mean(), dper.std(),
                  dideal.mean(), dideal.std(), m("FBnorm"), m("vn"),
                  100 * np.mean(q3 / q3d - 1), 100 * np.max(np.abs(q3 / q3d - 1))))
print("\n".join(out))
print("")
print("If mean(eb-pr) ~ 0 per trial but the earlier table showed -0.05, the gap was a mean-vs-mean artifact.")
print("If e/ideal != 1 per trial, the idealization err=(dn/cost)/sigma_min(G) is what fails (F-block geometry).")
