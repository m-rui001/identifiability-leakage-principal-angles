"""b63 -- what controls theta?  A designed sweep of the small-cosine ratio (round 32, main line 54.9).

Open item 1 as stated in the chapter: kappa is measured and beat = 1/2 kappa theta^2, so the whole
question is what controls theta = angle(v*, argmin_{|v|=1} lam_1(v)).  b60/b61/b62 measured theta on
random draws and could not say; b62 showed the residual there is the *evaluator's* resolution of g*,
not the size of theta.  This round therefore stops estimating coefficients and instead asks a question
with an EXACT input variable: the two smallest cosines of the dictionary.

Observation being turned into a hypothesis (from b60/b61's own tables, not from this script):
  spread 2-45-88  -> cosines 0.0349 (smallest) vs 0.707 (second smallest): ratio r = 0.049, theta = 0
                     on 9 of 10 draws (v* IS the sphere minimiser).
  spread 30-88-89 -> 0.01745 vs 0.0349: r = 0.50, theta in [3.4e-6, 8.7e-5] with the beat resolvable.
  spread 5-75     -> 0.0872 vs 0.1736: r = 0.50, theta ~ 3e-7 -- same r, far smaller theta.
So r alone cannot be the whole story, which is exactly what a designed sweep can separate: r is a
property of the SPECTRUM, while the top eigenvalue of A0 = K^T Lam^-1 K is a property of spectrum AND
subspace.  Two candidate mechanisms, two pre-registered directions.

OWN RNG STREAM: seed 61203, rebound in b57's namespace before any draw.  No cross-script per-draw
comparison is implied (rule: different stream => only within-run comparisons).

DESIGN (deterministic in the swept quantity, Monte-Carlo only in Lam and in K):
  family B  cosines = (0.90, 0.50, r*0.50)   r over geomspace(1e-4, 0.999, 8)
            -- the two SMALLEST cosines approach each other; the top cosine stays at 0.90.
  family T  cosines = (0.90, 0.90-d, 0.40)   d over geomspace(1e-4, 0.4, 8)
            -- the two LARGEST approach each other; the small end stays put.  Confound control.
  5 draws per (family, r): Lam = lam_spd(6), K = blocks(Lam, cosines) as always.
  theta measured from b60's 6-start sphere descent (the same estimator that produced (viii));
  no 200k grid here -- the grid only upper-bounds the infimum and is not needed for an angle.

FALSIFIERS (directions fixed before any number exists):
  M1  family B: theta is NOT non-decreasing in r (Spearman sign flips, or the top-r theta is below the
      bottom-r theta by more than the measured per-draw floors).  If it fires, the "small-cosine
      degeneracy" reading of open item 1 is dead and I say so.
  M2  family T: theta GROWS with the top-side closeness.  If it fires, the mechanism is "near-degeneracy
      anywhere in the fibre", not "at the small-cosine end", and the wording of (viii)/(ix) has to change.
      Predicted NOT to fire, because Lam^-1 weights the small cosines.
  M3  leverage gate (rule 17, this is a precondition not a result): a draw is testable only if
      |grad_sphere(v_b)| <= 1e-6*|lam_1(v_b)| AND beat > 10 floors in this draw's own gap-relative
      units.  Per r I report #testable/5.  If fewer than half the r-values have >=3 testable draws,
      the monotonicity test was NOT RUN and I report that, not a pass.
  M4  DIAGNOSTIC ONLY (cannot support a claim at 5 draws per r): does theta track the intrinsic ratio
      rho_A = lam_1(A0)/lam_2(A0) better than it tracks the input r?  Printed so that a future round
      can decide which variable to sweep; explicitly not evidence.
  M5  bookkeeping: theta must be 0 when the descent returns v_b with |v_b . v*| = 1, and the beat must
      be >= 0 on every draw (b60's F1/F2 direction).  A negative beat here means the descent is broken.
  NOT registered: the size of theta, the deficit over lambda_min(G), anything about kappa.
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ns = {}
_src = open("b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)
_ns["rng"] = np.random.default_rng(61203)

lam_spd = _ns["lam_spd"]
blocks = _ns["blocks"]
setup = _ns["setup"]
rng = _ns["rng"]

_bns = {}
_bns["__file__"] = "b60_sphere_gradient.py"
_b60 = open("b60_sphere_gradient.py", encoding="utf-8").read()
exec(compile(_b60.split("CELLS = [")[0], "b60-head", "exec"), _bns)
_ns["rng"] = np.random.default_rng(61203)          # keep b60's head from rebinding my stream
lam_of_v = _bns["lam_of_v"]
grad_sphere = _bns["grad_sphere"]
sphere_descent = _bns["sphere_descent"]
lam_roots = _bns["lam_roots"]
crossings_stable = _bns["crossings_stable"]

RS = np.geomspace(1e-4, 0.999, 8)
DS = np.geomspace(1e-4, 0.40, 8)
FAM = [("B (small end)", [(0.90, 0.50, r * 0.50) for r in RS], RS),
       ("T (large end)", [(0.90, 0.90 - d, 0.40) for d in DS], DS)]
NDRAW = 5

print("=" * 132)
print("[S0] evaluator anchors on this run (no new algebra, just the sticks I read with)")
print("=" * 132)
g2, g1 = [], []
rec = {}
for fam, cos_list, swept in FAM:
    for cv, x in zip(cos_list, swept):
        for _d in range(NDRAW):
            Lam = lam_spd(6)
            s = setup(Lam, blocks(Lam, list(cv)))
            if s is None:
                rec.setdefault((fam, x), []).append(None)
                continue
            gap = s["cf1"] - s["true"]
            g2.append(abs(lam_of_v(s, s["v1"])[0] - s["cf1"]))
            lamv, Bv, Cv = lam_of_v(s, s["vstar"])
            g1.append(abs(lamv - lam_roots(Bv, Cv)))

            starts = [s["v1"], s["vstar"], np.linalg.eigh(s["B2"])[1][:, -1],
                      rng.normal(size=s["A0"].shape[0]), rng.normal(size=s["A0"].shape[0]),
                      rng.normal(size=s["A0"].shape[0])]
            res = [sphere_descent(s, w0) for w0 in starts]
            lam_b, v_b = min(res, key=lambda r: r[0])[:2]
            gn_b = float(np.linalg.norm(grad_sphere(s, v_b)[1]))
            th = float(np.arccos(min(1.0, abs(float(s["vstar"] @ v_b)))))
            beat = lamv - lam_b
            a_b, B_b, C_b = lam_of_v(s, v_b)
            fl = max(abs(a_b - lam_roots(B_b, C_b)), abs(lamv - lam_roots(Bv, Cv)))
            w0A = np.linalg.eigvalsh(s["A0"])
            rho_A = float(w0A[-1] / w0A[-2]) if w0A[-2] > 0 else np.nan
            testable = (gn_b <= 1e-6 * abs(lam_b)) and (beat > 10.0 * fl)
            rec.setdefault((fam, x), []).append(dict(gap=gap, th=th, beat=beat, fl=fl,
                                                     beat_g=beat / gap, th_fl=np.nan, gn=gn_b,
                                                     rho=rho_A, test=testable, defc=(lam_b - s["true"]) / gap))
print(f"  |lam_1(v1)-cf_1| max = {max(g2):.3e} (gate 1e-15) ; |lam_-(v*) - np.roots| max = {max(g1):.3e}"
      f" (gate 1e-13)  -> {'PASS' if max(g2) <= 1e-15 and max(g1) <= 1e-13 else 'VOID'}")

print("\n" + "=" * 132)
print("[S1] the sweep.  th = angle(v*, sphere min); beat/gap = lambda_1(v*) minus that value over gap;")
print("     #test = draws passing M3 (|grad(v_b)|<=1e-6|lam| and beat>10 floors); rho_A = lam1/lam2 of A0.")
print("=" * 132)
for fam, _cl, swept in FAM:
    print(f"  --- family {fam}: swept = {swept[0]:.1e} .. {swept[-1]:.3g}")
    print(f"  {'swept':>10} {'th med':>10} {'th max':>10} {'beat/gap med':>13} {'beat/gap max':>13} "
          f"{'#test':>6} {'|grad(v_b)| max':>16} {'rho_A med':>10} {'deficit med':>12}")
    for x in swept:
        a = [r for r in rec[(fam, x)] if r is not None]
        if not a:
            print(f"  {x:>10.3e}   (all draws rejected by setup)")
            continue
        ths = np.array([r["th"] for r in a])
        bg = np.array([r["beat_g"] for r in a])
        nt = sum(1 for r in a if r["test"])
        print(f"  {x:>10.3e} {np.median(ths):>10.3e} {ths.max():>10.3e} {np.median(bg):>13.3e} "
              f"{bg.max():>13.3e} {nt:>6}/{len(a)} {max(r['gn'] for r in a):>16.2e} "
              f"{np.median([r['rho'] for r in a]):>10.4f} {np.median([r['defc'] for r in a]):>12.4f}")

print("\n" + "=" * 132)
print("[S2] the falsifiers")
print("=" * 132)
for fam, _cl, swept in FAM:
    med, mx, nt, xs = [], [], [], []
    for x in swept:
        a = [r for r in rec[(fam, x)] if r is not None and r["test"]]
        if len(a) >= 3:
            xs.append(x)
            med.append(np.median([r["th"] for r in a]))
            mx.append(max(r["th"] for r in a))
            nt.append(len(a))
    med, mx = np.array(med), np.array(mx)
    flips = int(np.sum(np.diff(med) < 0.0)) if med.size else 0
    print(f"  {fam}: r-values with >=3 testable draws = {len(xs)}/{len(swept)}"
          f"   {'-> M3: test NOT RUN at these r (no claim possible)' if len(xs) < len(swept) else ''}")
    if len(xs) >= 4:
        spear = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(med)))[0, 1])
        print(f"     theta med over those r: {[f'{v:.2e}' for v in med]}")
        print(f"     Spearman(swept, theta med) = {spear:+.3f}   monotone non-decreasing = "
              f"{flips == 0}   (#sign flips = {flips})   "
              f"first-vs-last = {med[0]:.3e} -> {med[-1]:.3e}")
        if fam.startswith("B"):
            print(f"  M1 (family B must be non-decreasing in r): "
                  f"{'NOT fired' if flips == 0 and med[-1] >= med[0] else 'FIRED'}")
        else:
            print(f"  M2 (family T must stay ~0; growth would reword the mechanism): "
                  f"{'NOT fired' if med[-1] < 1e-3 * max(med[0], 1e-30) or med.max() < 1e-6 else 'FIRED'}"
                  f"   [max theta here = {mx.max():.3e}]")
negb = sum(1 for k, a in rec.items() for r in a if r is not None and r["beat"] < 0.0)
print(f"  M5 #(beat < 0) = {negb}   (must be 0; b60's F1 direction)")
allrec = [r for a in rec.values() for r in a if r is not None]
z = [r for r in allrec if r["th"] == 0.0]
print(f"  #(theta exactly 0) = {len(z)}/{len(allrec)}   #(theta < 1e-9) = "
      f"{sum(1 for r in allrec if r['th'] < 1e-9)}")
print(f"  M4 DIAGNOSTIC ONLY: theta tracks rho_A better than the swept quantity? "
      f"cannot be settled at {NDRAW} draws per point -- printed as ranges only:")
for fam, _cl, swept in FAM:
    pts = [(x, r["th"], r["rho"]) for x in swept for r in rec[(fam, x)] if r is not None]
    rr = np.array([p[2] for p in pts])
    print(f"     {fam}: rho_A range [{rr.min():.4f},{rr.max():.4f}]  "
          f"theta range [{min(p[1] for p in pts):.3e},{max(p[1] for p in pts):.3e}]")

print("\n" + "=" * 132)
print("判读（预登记，跑前写下的方向；跑后只填数，不改方向）")
print("  M3 又一次是**先决条件**：这次地板不再是 λ 的评测器精度（b62 在那儿死掉），而是")
print("     「`v_b` 真的是驻点」这件事 —— `|grad(v_b)| <= 1e-6|lam|`。这不严，因为它是我唯一")
print("     能给出的、与 gap 同尺度的判据（b60 的 F5 就是这么用的）。若每个 r 上 testable<3，")
print("     M1/M2 都算**没跑**，我按这句话写，不写成 0 违例。")
print("  M1 不触发（θ 随 r 单调不降）=> 「小余弦端的近重数是 θ 的来源」这条**第一次**有了")
print("     设计性证据，正文 (ix) 结尾的 open item 1 就能从『什么控制 θ』升级成")
print("     『θ 由纤维里最小的两个余弦之比控制』——注意这仍然是**观察级**，因为机制变量")
print("     可能是 ρ_A 而不是 r（M4），下一轮必须扫 ρ_A 才能分开代理与机制。")
print("  M1 触发 => 这条主线**第二次**判死（b62 是估计器、b63 是设计变量）。那 open item 1 我就")
print("     按 §54.9 彻底收尾：正文只保留 (viii)/(ix) 的上界陈述，我把精力转到认证侧。")
print("  M2 触发不丢人：机制只是从『小余弦端』变成『纤维里任何一处近重数』，正文措辞跟着改。")
print("     但**不许**在同一个文件里既宣布 M2 又把它当 M1 的支持 —— 两族扫的是不同的东西。")
