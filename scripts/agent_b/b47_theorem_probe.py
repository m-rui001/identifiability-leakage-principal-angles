"""
b47 -- theorem probe for b46.  TARGET CLAIM (T'):

  (T')  cf1 >= lam_min(G) UNCONDITIONALLY -- no equal-angle hypothesis, no cap case split,
        no resolution audit.  cf1 is a certified UPPER estimate of the smallest eigenvalue.

Derivation (and a first attempt at it that was WRONG, kept here because the probe is what caught it).
  For a unit vector v write  B(v) = v^T K^T Lam^-1 K v,  C(v) = v^T K^T Lam^-2 K v, so with the
  eigenpairs (l_i, u_i) of Lam and k_i = K^T u_i,
      phi_v(lam) := v^T K^T (Lam - lam I)^-1 K v = sum_i (k_i.v)^2/(l_i - lam)
                 = sum_i [(k_i.v)^2/l_i] * 1/(1 - lam/l_i).
  Let rho_v = sum_i (k_i.v)^2/l_i . delta_{1/l_i}: mass B(v), first moment C(v).  For
  0 < lam < l_1 the map y -> 1/(1 - lam y) is CONVEX on (0, 1/l_1), so Jensen's inequality gives
      phi_v(lam) >= B(v)^2/(B(v) - lam C(v)) =: s_v(lam)      for every unit v, every lam in [0,l_1).
  s_v is the CHORD surrogate of f(lam) = lam_max(K^T(Lam-lam I)^-1K) = max_v phi_v(lam):
  f(lam) >= s_v(lam).  Both s_v and f are increasing, 1-lam is decreasing, and s_v(0) = B(v) <= c^2
  while f(0) = c^2, so the crossing of s_v with the line 1-lam happens AT OR AFTER that of f:
      lam(v) := crossing of s_v with 1-lam  >=  lam*,   (1-lam*) = f(lam*),
  and since prop:multi says lam_min(G) = min(lbar, lam*) <= lam*, that is (T') as soon as some v is
  chosen in closed form.  THE CHOICE: v1 in E0 (top eigenspace of A(0)) maximising C, because then
  B(v1) = c^2 exactly and C(v1) = c^2 t1 by the definition t1 := lam_max(E0^T K^T Lam^-2 K E0)/c^2,
  and lam(v1) is the crossing of c^2/(1 - t1 lam) with 1-lam -- which is cf1 VERBATIM (same quadratic
  as eq:cf with t -> t1).  So cf1 >= lam* >= lam_min(G).  No other input was used.

  WHAT I GOT WRONG FIRST: I wrote "lower surrogate => earlier crossing", which is backwards; a LOWER
  bound on f pushes the crossing LATER, hence gives an UPPER bound on lam*.  The random-direction
  count below is what exposed it (200/200 directions sat ABOVE lam_min, none below).

  WHY cf FLIPS AND cf1 CANNOT: cf pairs intercept c^2 with slope c^2 t = lam_max(K^T Lam^-2 K), but
  the maximiser of B2 generally lies OUTSIDE E0, where B(v) < c^2; the curve c^2/(1-t lam) is then
  NOT of the form s_v for any v -- it dominates s_v in both intercept and slope -- so the Jensen
  bound does not cover it and its crossing may sit on either side of lam*.  Inflating the slope moves
  the crossing EARLIER, which is exactly the sign of b45's first-order law gap ~= -cf*slack/slope
  (slack = c^2(t - t1) >= 0).  The whole flip bias of cf was that one unlicensed replacement.

Pre-registered, before running:
  A1  min gap1 > 0 in every cell, 0 resolved flips (the claim itself, uncapped branch).
  A2  Jensen residual phi_v(lam) - s_v(lam) >= 0 at lam = 0.3/0.7/0.95 of cf1, for the v1 actually
      used, AND for 200 random unit vectors per draw.  Any resolved negative => convexity direction
      or the measure normalisation is wrong => (T') is dead.
  A3  family test: lambda(v) >= lam_min(G) for all 200 random v per draw, counted over ALL draws
      including the capped branch (the claim is unconditional, so the capped draws must pass too).
      Violations predicted: 0.  A single resolved violation kills (T').
  A4  cf <= cf1 in every draw where slack clears the floor, and coin-flip otherwise (E0 = whole space
      => t1 = t => cf1 = cf).  Report max|C(v1) - c^2 t1| to check the tie-break is the true max.
"""
import numpy as np

rng = np.random.default_rng(49377)
N = 100
EPS = np.finfo(float).eps
P = 4
NDIR = 200

SPECTRA = {
    "spread 5-75":     [np.cos(np.deg2rad(a)) for a in (5, 40, 75)],
    "spread 2-45-88":  [np.cos(np.deg2rad(a)) for a in (2, 45, 88)],
    "spread 30-88-89": [np.cos(np.deg2rad(a)) for a in (30, 88, 89)],
    "equal 20d":       [np.cos(np.deg2rad(20))] * 3,
}


def lam_spd(p):
    E = rng.normal(size=(100, p))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E.T @ E


def blocks(Lam, cosines):
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))
    M = Q1 @ np.diag(cosines) @ Q2.T
    ev, V = np.linalg.eigh(Lam)
    return (V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T) @ M


def cf_of(t, c2):
    s = 1.0 - c2
    disc = max((1.0 + t) ** 2 - 4.0 * s * t, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t)


def crossing(B, C):
    """lam solving B^2/(B - lam C) = 1 - lam, i.e. C lam^2 - (B+C) lam + B(1-B) = 0."""
    if C <= 0.0:
        return 1.0
    disc = (B + C) ** 2 - 4.0 * C * B * (1.0 - B)
    if disc < 0.0:                      # can only happen at round-off (B <= 1 forces disc >= 0)
        return 1.0
    return ((B + C) - np.sqrt(disc)) / (2.0 * C)


def jensen_min(Lam, K, v, B, C, cfl):
    """min over lam of phi_v(lam) - s_v(lam) at three test points; >= 0 is Jensen."""
    worst = np.inf
    for fr in (0.3, 0.7, 0.95):
        lam = fr * cfl
        phi = float(v @ K.T @ np.linalg.solve(Lam - lam * np.eye(Lam.shape[0]), K) @ v)
        worst = min(worst, phi - B * B / (B - lam * C))
    return worst


def probe(Lam, K):
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = np.linalg.eigvalsh(Gr)[0]
    A0 = K.T @ np.linalg.solve(Lam, K)
    B2 = K.T @ np.linalg.solve(Lam, np.linalg.solve(Lam, K))
    w0, V0 = np.linalg.eigh(A0)
    c2 = w0[-1]
    if c2 > 1.0 + 1e-12 or c2 <= 0.0:
        return None, "not PSD / degenerate"
    tol = 1e-9 * max(abs(c2), 1.0)
    E0 = V0[:, w0 >= w0[-1] - tol]
    t1 = float(np.linalg.eigvalsh(E0.T @ B2 @ E0)[-1])
    t = float(np.linalg.eigvalsh(B2)[-1])
    cf, cf1 = cf_of(t / c2, c2), cf_of(t1 / c2, c2)
    floor = 100.0 * EPS * np.linalg.norm(Gr, 2)
    v1 = E0[:, -1]                                    # unit, in E0, maximises C over E0
    B1, C1 = float(v1 @ A0 @ v1), float(v1 @ B2 @ v1)
    out = dict(true=true, cf=cf, cf1=cf1, floor=floor, t=t, t1=t1, c2=c2, m0=E0.shape[1],
               gap1=cf1 - true, gap=cf - true,
               Jv1=jensen_min(Lam, K, v1, B1, C1, cf1),
               dev=abs(C1 - t1))
    R = rng.normal(size=(NDIR, q))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    Bs = np.einsum("ij,ij->i", R @ A0, R)
    Cs = np.einsum("ij,ij->i", R @ B2, R)
    ls = np.array([crossing(b, c) for b, c in zip(Bs, Cs)])
    out["Jrand"] = min(jensen_min(Lam, K, R[i], Bs[i], Cs[i], cf1) for i in range(NDIR)
                       if Cs[i] > 0.0)
    out["fam_below_true"] = int(np.sum((ls < true) & (true - ls > floor)))
    out["fam_below_cf1"] = int(np.sum((ls < cf1) & (cf1 - ls > floor)))
    out["fam_min"] = float(ls.min())
    return out, "ok"


def mixgen(cs, delta):
    def g():
        K0 = blocks(np.eye(P), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = (U[:, 0] + U[:, 1]) / np.sqrt(2.0)
        Vv = np.column_stack([np.linalg.eigh(np.eye(P) - np.outer(weak, weak))[1][:, 1:4], weak])
        ev = np.ones(P)
        ev[-1] = 1.0 - delta
        Lam = Vv @ np.diag(ev) @ Vv.T
        evk = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
        return Lam, K0 * min(1.0, cs[0] / np.sqrt(evk[-1]))
    return g


def cell(tag, gen, keep_capped=False):
    rows = []
    for _ in range(N):
        Lam, K = gen()
        rec, _why = probe(Lam, K)
        if rec is None:
            continue
        lbar = min(np.linalg.eigvalsh(Lam)[0], 1.0)
        capped = not (rec["cf1"] < lbar and rec["true"] < lbar - 1e-10 * max(1.0, lbar))
        if capped and not keep_capped:
            continue
        rows.append(rec)
    if not rows:
        print(f"   {tag:>30}: all skipped")
        return rows
    g1 = np.array([r["gap1"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    res = int(np.sum(np.abs(g1) > fl))
    print(f"   {tag:>30}: n={len(rows):3d}  min gap1={g1.min():9.2e}  flips(res/n)="
          f"{int(np.sum((g1 < 0) & (np.abs(g1) > fl)))}/{res}  min(cf1-cf)="
          f"{min(r['cf1'] - r['cf'] for r in rows):9.2e}  worst J(v1)={min(r['Jv1'] for r in rows):9.2e}"
          f"  worst J(rand)={min(r['Jrand'] for r in rows):9.2e}  fam_below_true="
          f"{sum(r['fam_below_true'] for r in rows)}  fam_below_cf1={sum(r['fam_below_cf1'] for r in rows)}"
          f"  max|C1-t1|={max(r['dev'] for r in rows):9.2e}  max dim(E0)={max(r['m0'] for r in rows)}")
    return rows


print("=" * 118)
print("[A] TEST OF (T') cf1 >= lam_min(G), plus the two inequalities it rests on:")
print("    (i) Jensen: phi_v(lam) >= B^2/(B - lam C) for the v1 used and for 200 random v   [min >= 0]")
print("    (ii) the whole family lam(v) sits above lam_min(G), counted over ALL draws (A3)   [0 below]")
print("    (iii) cf <= cf1 (A4) and max|C(v1) - t1|, the tie-break check when E0 degenerates.")
for name, cs in SPECTRA.items():
    cell(name, lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(P)))
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "2-45-88 mix d=3e-2"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "30-88-89 mix d=3e-2"),
                      ([np.cos(np.deg2rad(2)), np.cos(np.deg2rad(84)), np.cos(np.deg2rad(88))], 3e-2,
                       "2d/84d/88d mix d=3e-2")]:
    cell(nm, mixgen(cs, delta))
for gm in (1.0, 0.6, 1.5):
    cell(f"gamma={gm} 2-45-88", lambda gm=gm: (lambda L: (L, blocks(L, SPECTRA["spread 2-45-88"])))
         (gm * np.eye(P)))
print("   [A3] unconditional version: nothing skipped, capped branch included.")
for name, cs in SPECTRA.items():
    cell(f"ALL {name}", lambda cs=cs: (lambda L: (L, blocks(L, cs)))(lam_spd(P)), keep_capped=True)
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 3e-2, "ALL 2-45-88 mix"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "ALL 30-88-89 mix")]:
    cell(nm, mixgen(cs, delta), keep_capped=True)

print("\n" + "=" * 118)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  A1 任一格 min gap1 出现 resolved 负值 => (T') 死，cf1 只能写回'经验上更好'。")
print("  A2 任一 resolved 负 Jensen 残差 => 凸性/归一化方向错，(T') 与整条推论作废，必须重推。")
print("  A3 fam_below_true 必须恒为 0（含 capped 分支）；这是 (T') 的全部内容，1 个反例即致命。")
print("  A4 min(cf1-cf) 必须 >= -地板；负到 resolved 说明 t1>t 的实现或 t1 的定义有错。")
print("     max|C1-t1| 非零说明我取的 v1 不是 E0 上 C 的最大者 => 命题要写成 sup/max over E0。")
print("  全部通过 => 正文新增一个无条件命题：cf1 是 lam_min(G) 的上界估计，且 cf 的翻转偏差被解释为")
print("     '用 E0 外的斜率冒充同一条弦的斜率'，一阶符号为负。")
