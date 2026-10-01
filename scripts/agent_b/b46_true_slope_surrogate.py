"""
b46 -- does cf get better if its pole is moved to the TRUE slope of f at 0?

Trigger.  b45 (community 42.2) identified cf as the [0/1] Padé -- the one-pole collapse -- of the
scalar function
      f(lam) = lam_max(K^T (Lam - lam I)^-1 K),      f(0) = c^2,
built with the pole at 1/t where c^2 t = ||Lam^-1 K||_2^2 = lam_max(K^T Lam^-2 K), whereas the TRUE
right-derivative of f at 0 is
      f'(0+) = lam_max(E0^T (K^T Lam^-2 K) E0)  =:  c^2 t1   <=  c^2 t,      t1 <= t,
E0 being the top eigenspace of K^T Lam^-1 K.  So cf uses an UPPER bound of the slope, and b45's
first-order law is exactly the statement that this replacement costs it
      D(lam) = f(lam) - c^2/(1 - t lam) = -lam*(c^2 t - c^2 t1) + o(lam),
i.e. the whole systematic flip bias of cf lives in (t - t1).

This script defines the corrected surrogate
      cf1 := cf(t1, alpha_top)      (same closed form, t -> t1; costs one extra q x q eigenproblem)
and asks the only question that matters for the chapter:  IS cf1 AN UPPER ESTIMATE OF
lam_min(Gr) WITH NO EQUAL-ANGLE HYPOTHESIS?  If yes, that is a strictly better certificate-side
object than prop:plane, at the same cost.  If no, cf's use of the upper slope was deliberate
robustness and I stop here.

Two things follow from the algebra and are recorded BEFORE running, so the prediction is not a
post-hoc fit:
  (A) cf1 >= cf always.  Q(lam,t) = (1-lam)(1-t lam) - c^2 = 0 gives
      dcf/dt = lam(1-lam)/(1 + t - 2 t lam) > 0 ... with a MINUS sign: Q_t = -lam(1-lam),
      Q_lam = -(1-t lam) - t(1-lam), so dlam/dt = -Q_t/Q_lam < 0.  Lowering t to t1 RAISES the root.
  (B) the shift is, to first order, EXACTLY the gap:
      cf1 - cf ~= cf(1-cf)(t-t1)/(c^2(1+t))   and   gap = cf - lam* ~= -cf*(c^2 t - c^2 t1)/slope,
      with slope = 1+f'(lam*).  In b45's flipping cells |R-1| <= 1.1e-4 means slope ~= 1, and there
      (1-cf)/(c^2(1+t)) ~= 1, so gap1 = gap + (cf1-cf) is expected to be SECOND order, i.e.
      |gap1|/|gap| should be small and NOT a fixed ratio -- the first-order term is gone by
      construction, so what remains is the o(lam) remainder of b45's law.

Pre-registered, before looking at any output:
  C1 (implementation control).  q = 1: A(0) is 1x1, so t1 == t EXACTLY and cf1 == cf to round-off;
      the flip counts must reproduce b45/B0 (0 resolved flips in 500 draws, gaps > 0).  Falsifier:
      any |cf1-cf| above round-off => I have implemented t1 wrong and nothing below means anything.
  C2 (the claim).  General Lam, five spectra: flips(cf1) == 0 among resolved draws.  ONE resolved
      draw with gap1 < 0 kills the claim, and then the only honest output is 'a lower-variance
      estimator', not 'an upper estimate'.
  C3 (the cells that flipped 100/100 in b45: multi-atom 'mix' geometry, the 2x2, both ladders).
      Same expectation, plus |gap1|/|gap| reported; predicted to be <= 0.1 and to vary.
  C4 (isotropy null).  Lam = gamma*I_p: t1 == t == 1/gamma, so cf1 == cf and stays exact (D1 at the
      eigvalsh floor).  Falsifier: a difference above round-off => the E0-restricted max is not the
      right derivative, i.e. b45's (ii) is wrong and prop:side must be re-checked.
  C5 (scaling).  If the residual is genuinely the second-order term, med|gap1| should fall like
      cf1^2 along b45/B6a's ladder (top cosine 2deg -> 30deg), i.e. log|gap1| vs log cf1 slope ~= 2.
      If the slope is ~= 1, cf1 still carries a first-order bias and I have not fixed anything.
"""
import numpy as np

rng = np.random.default_rng(48902)
N = 100
EPS = np.finfo(float).eps
P = 4

SPECTRA = {
    "equal 20d":       [np.cos(np.deg2rad(20))] * 3,
    "equal 5d":        [np.cos(np.deg2rad(5))] * 3,
    "spread 5-75":     [np.cos(np.deg2rad(a)) for a in (5, 40, 75)],
    "spread 2-45-88":  [np.cos(np.deg2rad(a)) for a in (2, 45, 88)],
    "spread 30-88-89": [np.cos(np.deg2rad(a)) for a in (30, 88, 89)],
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


def resolvent(Lam, K, lam):
    A = K.T @ np.linalg.solve(Lam - lam * np.eye(Lam.shape[0]), K)
    w, Vv = np.linalg.eigh(A)
    return w[-1], Vv[:, -1]


def cf_of(t, c2):
    """prop:plane closed form: smaller root of (1-lam)(1-t lam) = c2, with sin^2 alpha = 1-c2."""
    s = 1.0 - c2
    disc = max((1.0 + t) ** 2 - 4.0 * t * s, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t)


def diag2(Lam, K):
    """cf (pole at the upper slope) and cf1 (pole at the true slope), with both sign-law residuals."""
    p, q = Lam.shape[0], K.shape[1]
    Gr = np.block([[Lam, K], [K.T, np.eye(q)]])
    true = np.linalg.eigvalsh(Gr)[0]
    A0 = K.T @ np.linalg.solve(Lam, K)
    c2 = np.linalg.eigvalsh(A0)[-1]
    if c2 > 1.0 + 1e-12:
        return None, "Gr not PSD (top cosine > 1)"
    c2 = min(c2, 1.0)
    if c2 <= 0 or true <= 1e-12:
        return None, "degenerate"
    t = np.linalg.norm(np.linalg.solve(Lam, K), 2) ** 2 / c2
    w0, V0 = np.linalg.eigh(A0)
    E0 = V0[:, w0 >= w0[-1] - 1e-9 * max(abs(w0[-1]), 1.0)]
    B2 = K.T @ np.linalg.solve(Lam, np.linalg.solve(Lam, K))
    t1 = np.linalg.eigvalsh(E0.T @ B2 @ E0)[-1] / c2
    lbar = min(np.linalg.eigvalsh(Lam)[0], 1.0)
    floor = 100.0 * EPS * np.linalg.norm(Gr, 2)
    cf, cf1 = cf_of(t, c2), cf_of(t1, c2)
    if max(cf, cf1) >= lbar:
        return None, "cf beyond lbar"
    if true >= lbar - 1e-10 * max(1.0, lbar):
        return None, "capped branch"
    f_cf, _ = resolvent(Lam, K, cf)
    f_cf1, _ = resolvent(Lam, K, cf1)
    _, v_s = resolvent(Lam, K, true)
    d2 = np.linalg.solve(Lam - true * np.eye(p), K) @ v_s
    slope = 1.0 + d2 @ d2
    return dict(true=true, cf=cf, cf1=cf1, gap=cf - true, gap1=cf1 - true,
                D=f_cf - c2 / (1.0 - t * cf), D1=f_cf1 - c2 / (1.0 - t1 * cf1),
                slope=slope, floor=floor, t=t, t1=t1, c2=c2, lbar=lbar,
                slack=c2 * (t - t1), m0=E0.shape[1]), "ok"


def mism(g, d, fl):
    """sign-law mismatch among draws where BOTH sides clear the floor (the only readable count)."""
    both = (np.abs(g) > fl) & (np.abs(d) > fl)
    return int(np.sum((np.sign(g) * np.sign(d) < 0) & both)), int(both.sum())


def report(tag, rows, show_ratio=True):
    if not rows:
        print(f"   {tag:>32}: all skipped")
        return
    g = np.array([r["gap"] for r in rows]);  g1 = np.array([r["gap1"] for r in rows])
    d = np.array([r["D"] for r in rows]);    d1 = np.array([r["D1"] for r in rows])
    fl = np.array([r["floor"] for r in rows]); c1 = np.array([r["cf1"] for r in rows])
    m0r, n0 = mism(g, d, fl); m1r, n1 = mism(g1, d1, fl)
    print(f"   {tag:>32}: n={g.size:3d}  flips cf/cf1={int(np.sum((g<0)&(np.abs(g)>fl)))}/"
          f"{int(np.sum((g1<0)&(np.abs(g1)>fl)))}  med|gap| cf/cf1={np.median(np.abs(g)):9.2e}/"
          f"{np.median(np.abs(g1)):9.2e}  min gap1={g1.min():9.2e}"
          f"  mism1(res/n)={m1r}/{n1}  med|gap1/gap|={np.median(np.abs(g1[np.abs(g1)>fl])/np.abs(g[np.abs(g1)>fl])) if (np.abs(g1)>fl).any() else float('nan'):7.3f}"
          f"  med cf1={np.median(c1):7.4f}")


def run(gens, tag, n=N):
    rows = []
    for _ in range(n):
        Lam, K = gens()
        rec, _why = diag2(Lam, K)
        if rec is not None:
            rows.append(rec)
    report(tag, rows)
    return rows


def lam_with_weak(K, delta, weak):
    U = np.linalg.eigh(np.eye(P) - np.outer(weak, weak))[1][:, 1:4]
    Vv = np.column_stack([U, weak])
    ev = np.ones(P); ev[-1] = 1.0 - delta
    return Vv @ np.diag(ev) @ Vv.T


def mixgen(cs, delta):
    def g():
        K0 = blocks(np.eye(P), cs)
        U = np.linalg.svd(K0, full_matrices=True)[0]
        weak = (U[:, 0] + U[:, 1]) / np.sqrt(2.0)
        Lam = lam_with_weak(K0, delta, weak)
        ev = np.linalg.eigvalsh(K0.T @ np.linalg.solve(Lam, K0))
        return Lam, K0 * min(1.0, cs[0] / np.sqrt(ev[-1]))
    return g


def gen_lam(cs, lam_of):
    """lam_of takes p and returns a p x p SPD matrix; the lambda binds p = P."""
    return lambda: (lambda L: (L, blocks(L, cs)))(lam_of(P))


print("=" * 118)
print("[C1] implementation control: q=1 (single cosine).  t1 == t EXACTLY, so cf1 must equal cf and")
print("     the numbers must reproduce b45/B0 (0 resolved flips, gaps > 0).")
maxdiff = 0.0
for cval in [0.999, 0.99, 0.9, 0.7, 0.2]:
    rows = run(gen_lam([cval], lam_spd), f"cos={cval:5.3f}")
    if rows:
        maxdiff = max(maxdiff, max(abs(r["cf1"] - r["cf"]) for r in rows))
print(f"     max |cf1-cf| over q=1 cells = {maxdiff:.3e}   (predicted: 0 to round-off)")

print("\n" + "=" * 118)
print("[C2] general Lam (unit-norm Gram), five spectra.  CLAIM: flips(cf1) == 0 among resolved draws.")
for name, cs in SPECTRA.items():
    rows = run(gen_lam(cs, lam_spd), name)
    if rows:
        print(f"{'':>34}  t-1 range=[{min(r['t']-1 for r in rows):.3e},{max(r['t']-1 for r in rows):.3e}]"
              f"  t-t1 (=slack/c2) range=[{min(r['t']-r['t1'] for r in rows):.3e},"
              f"{max(r['t']-r['t1'] for r in rows):.3e}]  cf1>=cf in {sum(r['cf1']>=r['cf'] for r in rows)}/{len(rows)}")

print("\n" + "=" * 118)
print("[C3] the cells that flipped 100/100 in b45: multi-atom 'mix' geometry, b45's 2x2, both ladders.")
for cs, delta, nm in [(SPECTRA["spread 2-45-88"], 1e-2, "2-45-88 mix d=1e-2"),
                      (SPECTRA["spread 2-45-88"], 3e-2, "2-45-88 mix d=3e-2"),
                      (SPECTRA["spread 30-88-89"], 3e-2, "30-88-89 mix d=3e-2"),
                      ([np.cos(np.deg2rad(2)), np.cos(np.deg2rad(84)), np.cos(np.deg2rad(88))], 3e-2,
                       "top .9994/2nd .105 mix"),
                      ([np.cos(np.deg2rad(30)), np.cos(np.deg2rad(45)), np.cos(np.deg2rad(88))], 3e-2,
                       "top .866/2nd .816 mix"),
                      ([np.cos(np.deg2rad(30)), np.cos(np.deg2rad(84)), np.cos(np.deg2rad(88))], 3e-2,
                       "top .866/2nd .121 mix (b45 no-flip)")]:
    run(mixgen(cs, delta), nm)

print("\n" + "=" * 118)
print("[C4] isotropy null: Lam = gamma*I_p must give t1 == t == 1/gamma, cf1 == cf, exact (D1 at floor).")
for gm in (1.0, 1.5, 0.6):
    for name in ("equal 20d", "spread 2-45-88"):
        rows = run(gen_lam(SPECTRA[name], lambda p: gm * np.eye(p)), f"gamma={gm} {name}")
        if rows:
            print(f"{'':>34}  max|t1-1/gamma|={max(abs(r['t1']-1.0/gm) for r in rows):.3e}"
                  f"  max|cf1-cf|={max(abs(r['cf1']-r['cf']) for r in rows):.3e}"
                  f"  max|D1|={max(abs(r['D1']) for r in rows):.3e}")

print("\n" + "=" * 118)
print("[C5] scaling along the top-cosine ladder (second cosine held near 0.71, mix geometry): if cf1's")
print("     residual is the second-order term, log|gap1| vs log(cf1) should have slope ~= 2, not 1.")
pts = []
for adeg in (2, 3, 5, 8, 12, 20, 30):
    cs = [np.cos(np.deg2rad(adeg)), np.cos(np.deg2rad(45)), np.cos(np.deg2rad(88))]
    rows = run(mixgen(cs, 3e-2), f"ladder top={adeg}d")
    if rows:
        pts.append((np.median([r["cf1"] for r in rows]), np.median(np.abs([r["gap1"] for r in rows]))))
lg = np.log(np.array(pts))
print(f"     log-log slope of |gap1| vs cf1 = {np.polyfit(lg[:,0], lg[:,1], 1)[0]:.3f}"
      f"   (2 = first-order bias removed, 1 = still biased)")

print("\n" + "=" * 118)
print("判读（预登记，跑前写的；实测判读附在文件末尾）：")
print("  C1 |cf1-cf| 必须在地板以下，否则 t1 的实现是错的，后面全部作废。")
print("  C2/C3 flips(cf1) 只要出现 1 个 resolved 反例 => 'cf1 是无等角假设的上界估计' 死亡，")
print("     只准写成'方差更小的估计'；全 0 => 正文 prop:plane 的上界侧可以推广到一般 Lam（新结论）。")
print("  C3 |gap1/gap| 若 ~1（没有压缩）=> 我的 (B) 一阶相消推导错，gap 的主体不是 slack 项。")
print("  C4 若 gamma != 1 时 t1 != 1/gamma => b45 的 (ii)/cor:iso 里 E0-max 不是右导数，必须再撤。")
print("  C5 斜率 ~= 2 => 余项确实二阶；~= 1 => cf1 仍带一阶偏置，'修正'无效，按 §41.8 规矩停手。")
