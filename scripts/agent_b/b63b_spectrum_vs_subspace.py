"""b63b -- after b63 falsified both registered directions: is it the SPECTRUM or the SUBSPACE?

b63 (`b63_theta_small_cosine_sweep.txt`) ran with leverage on every swept point (M3: 5/5 testable, all
|grad(v_b)| <= 1.7e-8), and BOTH registered predictions fired:
  M1  theta is NOT monotone in the ratio of the two smallest cosines (Spearman +0.548, 3 sign flips);
  M2  theta is LARGER when the two LARGEST cosines coincide (4.6e-3 at d=1e-4 vs 3.1e-4 at d=0.40),
      the opposite of what I wrote in the docstring as the a-priori reason ("Lam^-1 weights the small
      cosines").  That reason was wrong, and b63b starts from the exact identity that exposes why.

THE IDENTITY I CHECK FIRST (a constructed object's defining relation, before any consequence):
  blocks() builds K = Lam^{1/2} M with M = Q1 diag(cosines) Q2^T, hence
      A0 = K^T Lam^{-1} K = M^T M          (Lam cancels COMPLETELY),
      B2 = K^T Lam^{-2} K = M^T Lam^{-1} M (Lam survives, weighting the SMALL cosines).
  So the top of spec(A0) -- which is what E0 and therefore v1 are built from -- is governed by the
  LARGEST cosines, and the small-cosine end enters only through B2.  That is the opposite of my b63
  prediction and it is verified numerically below to a stated tolerance before I use it.

WHAT b63 CANNOT SETTLE: in family B, rho_A = lam_1(A0)/lam_2(A0) is constant (3.2400) while theta spans
2.63e-6 .. 6.86e-4, and in family T rho_A and the swept gap move together.  So neither "the small-cosine
ratio" nor "rho_A" is separated from the draw of the subspace (Q1, Q2).

DESIGN (the confound breaker): hold the SPECTRUM fixed and sweep the SUBSPACE.  rho_A then varies
across draws on its own, so theta vs rho_A is an within-spectrum comparison, and the two spectra differ
only in whether the top cosines coincide.
  spectrum DEGEN = (0.90, 0.90 - 1e-4, 0.40)   (top pair coincides)
  spectrum SPACED = (0.90, 0.50, 0.40)          (nothing coincides)
  40 draws each: Lam = lam_spd(6), K = blocks(Lam, cosines); v*, the 6-start sphere descent, theta,
  beat/gap, rho_A, and the deficit over lambda_min(G).

FALSIFIERS (directions written before the numbers exist):
  S0  gate: |A0 - M^T M| <= 1e-12 and |B2 - M^T Lam^{-1} M| <= 1e-12 on every draw; a failure voids the
      mechanism story I am about to tell (and says my reading of blocks() is wrong, not the data).
  S1  spectrum control: median theta(DEGEN) is NOT above median theta(SPACED) by a factor >= 2, or
      min theta(DEGEN) > max theta(SPACED) fails -- reported as the exact comparison, not as a p-value.
      If S1 fires, "the top-cosine coincidence controls theta" is dead too and open item 1 gets closed
      as "no single input quantity I can name controls it".
  S2  subspace control within a fixed spectrum: the theta range over 40 draws spans < 1 order of
      magnitude.  If it fires, the draw of K is irrelevant and a spectrum-only statement could live.
      Predicted NOT to fire (b63's fixed-rho_A spread of 2.6 orders says K matters).
  S3  rho_A as the mechanism: WITHIN each spectrum, theta is non-increasing in rho_A (Spearman >= -0.5
      and no sign flip between the lowest and highest rho_A quartiles' medians).  Predicted to hold in
      DEGEN only if rho_A is the right variable.  If it fires, rho_A is demoted to a correlate.
  S4  leverage (rule 17): a draw counts only if |grad_sphere(v_b)| <= 1e-6|lam_1(v_b)| AND
      beat > 10 floors; per spectrum I report #testable/40 and use only those in S1/S3.
  S5  bookkeeping: beat >= 0 and deficit >= 0 on every draw (b60's F1/F2 directions).
  NOT registered: the size of theta, of the deficit, of rho_A.
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

_ns = {}
_src = open("b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(_src.split("def run_cell(")[0], "b57-head", "exec"), _ns)
_ns["rng"] = np.random.default_rng(61204)

lam_spd = _ns["lam_spd"]
setup = _ns["setup"]
rng = _ns["rng"]

_bns = {}
_bns["__file__"] = "b60_sphere_gradient.py"
_b60 = open("b60_sphere_gradient.py", encoding="utf-8").read()
exec(compile(_b60.split("CELLS = [")[0], "b60-head", "exec"), _bns)
lam_of_v = _bns["lam_of_v"]
grad_sphere = _bns["grad_sphere"]
sphere_descent = _bns["sphere_descent"]
lam_roots = _bns["lam_roots"]

# blocks() re-implemented here ONLY to keep M for the S0 identity check; identical construction.
def blocks_with_M(Lam, cosines):
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))
    M = Q1 @ np.diag(cosines) @ Q2.T
    ev, V = np.linalg.eigh(Lam)
    return (V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T) @ M, M

SPECS = {"DEGEN (0.90, 0.90-1e-4, 0.40)": (0.90, 0.90 - 1e-4, 0.40),
         "SPACED (0.90, 0.50, 0.40)": (0.90, 0.50, 0.40)}
NDRAW = 40

rec = {k: [] for k in SPECS}
s0a, s0b, g2, g1 = [], [], [], []
for name, cv in SPECS.items():
    for _d in range(NDRAW):
        Lam = lam_spd(6)
        K, M = blocks_with_M(Lam, list(cv))
        s = setup(Lam, K)
        if s is None:
            rec[name].append(None)
            continue
        Li = np.linalg.inv(Lam)
        s0a.append(float(np.max(np.abs(s["A0"] - M.T @ M))))
        s0b.append(float(np.max(np.abs(s["B2"] - M.T @ Li @ M))))
        gap = s["cf1"] - s["true"]
        g2.append(abs(lam_of_v(s, s["v1"])[0] - s["cf1"]))
        lamv, Bv, Cv = lam_of_v(s, s["vstar"])
        # D2 (post-run addition, registers nothing): is v* a CRITICAL POINT of the two-moment functional?
        # grad_sphere returns (lam, projected grad, B, C, sqrt(disc)); the rounding scale for the gradient
        # is DEFINED here as 100*eps*(|2 lam_B| |A0|_2 + |2 lam_C| |B2|_2), a bound on the terms summed,
        # and it is stated as a defined relative scale, not as an empirically measured floor.
        _l, _gs, _B, _C, _sq = grad_sphere(s, s["vstar"])
        _Fl = 2.0 * _C * _l - _B - _C
        _lB = -(1.0 - _l - 2.0 * _B) / _Fl
        _lC = -(_l * _l - _l) / _Fl
        gn_star = float(np.linalg.norm(_gs))
        fg_star = 100.0 * np.finfo(float).eps * (abs(2.0 * _lB) * np.linalg.norm(s["A0"], 2)
                                                 + abs(2.0 * _lC) * np.linalg.norm(s["B2"], 2))
        g1.append(abs(lamv - lam_roots(Bv, Cv)))
        starts = [s["v1"], s["vstar"], np.linalg.eigh(s["B2"])[1][:, -1],
                  rng.normal(size=s["A0"].shape[0]), rng.normal(size=s["A0"].shape[0]),
                  rng.normal(size=s["A0"].shape[0])]
        lam_b, v_b = min([sphere_descent(s, w0) for w0 in starts], key=lambda r: r[0])[:2]
        gn_b = float(np.linalg.norm(grad_sphere(s, v_b)[1]))
        th = float(np.arccos(min(1.0, abs(float(s["vstar"] @ v_b)))))
        beat = lamv - lam_b
        a_b, B_b, C_b = lam_of_v(s, v_b)
        fl = max(abs(a_b - lam_roots(B_b, C_b)), abs(lamv - lam_roots(Bv, Cv)))
        wA = np.linalg.eigvalsh(s["A0"])
        rec[name].append(dict(gap=gap, th=th, beat=beat, bg=beat / gap, rho=float(wA[-1] / wA[-2]),
                              defc=(lam_b - s["true"]) / gap, gn_star=gn_star, fg_star=fg_star,
                              test=(gn_b <= 1e-6 * abs(lam_b)) and (beat > 10.0 * fl), gn=gn_b))

print("=" * 132)
print("[S0] gates on the mechanism identity and on the evaluator")
print("=" * 132)
print(f"  max |A0 - MtM| = {max(s0a):.3e}   max |B2 - Mt Lam^-1 M| = {max(s0b):.3e}   (gate 1e-12)"
      f"   -> {'PASS: Lam cancels out of A0 exactly as claimed' if max(max(s0a), max(s0b)) <= 1e-12 else 'VOID'}")
print(f"  |lam_1(v1)-cf_1| max = {max(g2):.3e} (1e-15); |lam_-(v*) - np.roots| max = {max(g1):.3e}"
      f" (1e-13)  -> {'PASS' if max(g2) <= 1e-15 and max(g1) <= 1e-13 else 'VOID'}")

print("\n" + "=" * 132)
print("[S1]/[S2] theta over 40 draws of the SUBSPACE at a FIXED spectrum")
print("=" * 132)
print(f"  {'spectrum':>34} {'#test':>7} {'rho_A range':>20} {'theta min/med/max':>34} "
      f"{'beat/gap med':>13} {'deficit med':>12}")
tests = {}
for name in SPECS:
    a = [r for r in rec[name] if r is not None]
    t = [r for r in a if r["test"]]
    tests[name] = t
    th = np.array([r["th"] for r in t]) if t else np.array([np.nan])
    print(f"  {name:>34} {len(t):>3}/{len(a)} {min(r['rho'] for r in a):>9.4f}-{max(r['rho'] for r in a):<9.4f}"
          f" {th.min():>10.3e} /{np.median(th):<9.3e} /{th.max():<10.3e} "
          f"{np.median([r['bg'] for r in t]):>13.3e} {np.median([r['defc'] for r in t]):>12.4f}")
d = tests["DEGEN (0.90, 0.90-1e-4, 0.40)"]
p = tests["SPACED (0.90, 0.50, 0.40)"]
md, mp = np.median([r["th"] for r in d]), np.median([r["th"] for r in p])
print(f"  S1 spectrum control: median theta DEGEN/SPACED = {md:.3e}/{mp:.3e} = "
      f"{md / mp:.3f}x   (need >=2) ; min DEGEN {min(r['th'] for r in d):.3e} vs max SPACED "
      f"{max(r['th'] for r in p):.3e}  -> {'HOLDS' if md >= 2 * mp else 'S1 FIRED: no spectrum control'}")
for name, t in tests.items():
    th = np.array([r["th"] for r in t])
    print(f"  S2 subspace spread within {name}: {th.min():.3e}..{th.max():.3e} = "
          f"{np.log10(th.max() / max(th.min(), 1e-300)):.2f} decades  -> "
          f"{'K IS IRRELEVANT' if np.log10(th.max() / max(th.min(), 1e-300)) < 1 else 'the draw of K matters'}")

print("\n" + "=" * 132)
print("[S3] is rho_A the mechanism?  WITHIN each fixed spectrum, theta vs rho_A across subspace draws")
print("=" * 132)
print("  [POST-RUN ADDITION -- see IMPLEMENTATION HISTORY] a Spearman against a regressor that cannot vary")
print("  within a group is void, not a result, so the within-group RANGE of rho_A is now printed FIRST and")
print("  the correlation verdict is only issued when that range exceeds float noise (1e-12 relative).")
for name, t in tests.items():
    if len(t) < 8:
        print(f"  {name}: only {len(t)} testable draws -- NOT RUN")
        continue
    rho = np.array([r["rho"] for r in t]); th = np.array([r["th"] for r in t])
    rng_spread = float(rho.max() - rho.min())
    o = np.argsort(rho)
    k = len(t) // 4
    q1m, q4m = np.median(th[o[:k]]), np.median(th[o[-k:]])
    sp = float(np.corrcoef(np.argsort(np.argsort(rho)), np.argsort(np.argsort(th)))[0, 1])
    print(f"  {name}: within-group rho_A max-min = {rng_spread:.3e}  (relative {rng_spread / rho.mean():.3e})"
          f"   theta spread = {np.log10(th.max() / th.min()):.2f} decades")
    if rng_spread <= 1e-12 * rho.mean():
        print(f"    -> rho_A is CONSTANT (to float noise) while theta varies {np.log10(th.max() / th.min()):.2f}"
              f" decades: no function of spec(A0) can explain the within-spectrum spread.  The Spearman"
              f" ({sp:+.3f}) printed above would be correlation against rounding noise -> S3 SPEARMAN VOID;"
              f" the invariance statement itself is the finding.")
    else:
        print(f"  {name}: Spearman(rho_A, theta) = {sp:+.3f}   lowest-quartile med theta = {q1m:.3e}"
              f"   highest-quartile med = {q4m:.3e}   ratio low/high = {(q1m / q4m if q4m > 0 else float('nan')):.3f}"
              f"  -> {'consistent with rho_A control' if sp <= -0.5 and q1m > q4m else 'S3 FIRED: rho_A is only a correlate'}")

print("\n" + "=" * 132)
print("[S4]/[S5] leverage and bookkeeping  (run exactly as pre-registered)")
print("=" * 132)
allr = [r for a in rec.values() for r in a if r is not None]
print(f"  #(testable) = {sum(1 for r in allr if r['test'])}/{len(allr)}   "
      f"#(|grad(v_b)|>1e-6|lam|) = {sum(1 for r in allr if r['gn'] > 1e-6)}")
print(f"  S5 #(beat < 0) = {sum(1 for r in allr if r['beat'] < 0)}   "
      f"#(deficit < 0) = {sum(1 for r in allr if r['defc'] < 0)}   "
      f"min deficit = {min(r['defc'] for r in allr):.4f}")

print("\n" + "=" * 132)
print("[S6] post-run addition: the exact identity spec(A0) = {cos_i^2} (a separate RNG; it does not touch")
print("     the S1-S5 sample above, and no claim is registered on it -- it is a definitional check)")
print("=" * 132)
_s6 = []
_ns["rng"] = rng = np.random.default_rng(61205)   # S6 draws from its OWN stream, not the S1-S5 one
for name, cv in SPECS.items():
    for _ in range(12):
        Lam = lam_spd(6)
        K, M = blocks_with_M(Lam, list(cv))
        s = setup(Lam, K)
        if s is None:
            continue
        wA = np.sort(np.linalg.eigvalsh(s["A0"]))[::-1]
        wc = np.sort(np.array(cv) ** 2)[::-1]
        _s6.append(float(np.max(np.abs(wA - wc))))
        _s6.append(float(abs(wA[0] / wA[1] - (max(cv) / sorted(cv)[-2]) ** 2)))
print(f"  max over 24 samples of |sorted eig(A0) - cos^2| and of |rho_A - (c1/c2)^2| = {max(_s6):.3e}"
      f"  -> {'PASS: spec(A0) is EXACTLY the squared cosines, independent of Lam, Q1, Q2' if max(_s6) <= 1e-12 else 'FAIL'}")
print(f"  consequence: rho_A = (c_1/c_2)^2, a function of the cosines ALONE, so by construction it cannot")
print(f"  vary at fixed spectrum -- which is exactly why S3's within-group correlation had no regressor.")
print(f"  printed values check: DEGEN (0.90/0.8999)^2 = {(0.90 / (0.90 - 1e-4)) ** 2:.4f} (S1 row said 1.0002);"
      f" SPACED (0.90/0.50)^2 = {(0.90 / 0.50) ** 2:.4f} (S1 row said 3.2400).")

print("\n" + "=" * 132)
print("[DIAGNOSTIC -- NOT REGISTERED, cannot be quoted as a claim] is theta a VALUE error or a FLATNESS?")
print("=" * 132)
print("  references in this file, stated once because they are NOT the same quantity:")
print("    lam_of_v(v) = the two-moment (A0,B2) functional's smallest root for that direction;")
print("    'true'      = the EXACT lambda_min(G) of the 2p x 2p block matrix;")
print("    gap         = cf_1 - true  (the certificate's slack);   beat = lam_of_v(v*) - lam_of_v(v_b);")
print("    deficit     = (lam_of_v(v_b) - true) / gap.  So beat/gap is 'how far the descent undershoots the")
print("    functional AT v*', in gap units' -- it is not a statement about lambda_min(G).")
for name, t in tests.items():
    bg = np.array([r["bg"] for r in t]); mth = np.array([r["th"] for r in t])
    sp = float(np.corrcoef(np.argsort(np.argsort(bg)), np.argsort(np.argsort(mth)))[0, 1])
    print(f"  {name}: beat/gap min/med/max = {bg.min():.3e} /{np.median(bg):.3e} /{bg.max():.3e}"
          f"   (spread {bg.max() / bg.min():.1f}x = {np.log10(bg.max() / bg.min()):.2f} decades)"
          f"  at theta min/med/max = {mth.min():.3e} /{np.median(mth):.3e} /{mth.max():.3e}"
          f"   Spearman(theta, beat) WITHIN group = {sp:+.3f}")
    print(f"     implied curvature if beat = 1/2 kappa theta^2: kappa/gap = {2 * np.median(bg) / np.median(mth) ** 2:.3e}")
print("  CORRECTING THE SENTENCE I DRAFTED BEFORE LOOKING AT THE SPREAD: the two MEDIAN beat/gap values agree")
print("  to 1.3% (6.908e-05 vs 6.825e-05), but per draw beat/gap spans ~4.9 (DEGEN) and ~3.6 (SPACED)")
print("  decades, so 'the descent reproduces the value at v* to ~7e-5 gap in EVERY draw' is FALSE and is")
print("  withdrawn here.  What survives is only a median-level coincidence; whether it carries directional")
print("  information is exactly what the within-group Spearman(theta, beat) printed above decides, and the")
print("  verdict is filled in post-run.  Any curvature-based registration would have to measure kappa")
print("  itself rather than infer it from beat -- and all of this stays a diagnostic because none of it was")
print("  pre-registered before the run.")

print("\n" + "=" * 132)
print("[D2] post-run addition, registers nothing: is v* a CRITICAL POINT of the two-moment functional?")
print("=" * 132)
print("  If ||grad_S lam_-(v*)|| is far above its rounding scale, then v* is NOT a stationary point of the")
print("  functional I descend, and a small theta displaces lam_-(v*) only to FIRST order:")
print("    lam_-(v*) - lam_-(v_b) ~ |grad| * theta * cos(angle between -grad and the displacement),")
print("  which would make beat and theta ONE measurement, not two -- and it is the same mechanism that made")
print("  b62's leverage track |g*|.  Discriminator: slope of log beat on log theta (1 = linear, 2 = quadratic).")
for name, t in tests.items():
    gs = np.array([r["gn_star"] for r in t]); fs = np.array([r["fg_star"] for r in t])
    th = np.array([r["th"] for r in t]); bt = np.array([r["beat"] for r in t])
    sl, _ic = np.polyfit(np.log(th), np.log(bt), 1)
    sd1 = float(np.std(np.log(bt) - np.polyval([sl, _ic], np.log(th))))
    print(f"  {name}: ||grad(v*)|| min/med/max = {gs.min():.3e} /{np.median(gs):.3e} /{gs.max():.3e}"
          f"   ratio to its rounding scale: min = {np.min(gs / fs):.1f}x   med = {np.median(gs / fs):.1f}x")
    print(f"     #(||grad(v*)|| <= 10x rounding scale) = {int(np.sum(gs <= 10 * fs))}/{len(t)}"
          f"   ->  {'v* IS effectively stationary' if np.sum(gs <= 10 * fs) == len(t) else 'v* is NOT a critical point of the functional'}")
    print(f"     slope of log beat on log theta = {sl:+.3f}   (residual geometric sd {sd1:.3f});"
          f"  beat/theta med = {np.median(bt / th):.3e}  beat/theta^2 med = {np.median(bt / th ** 2):.3e}")
print("  For comparison, b62's own reading was theta ~ g*/kappa* -- a FIRST-order relation, consistent with")
print("  a non-stationary v*; the numbers above say whether that also holds in b63b's samples.")
print("  [D2 SELF-CORRECTION, written after seeing the slopes] two things, in order:")
print("  (a) the header reasoned as if the Taylor expansion were about v*, but beat = lam_-(v*) - lam_-(v_b)")
print("      and v_b IS a critical point of the functional (S4: ||grad(v_b)|| <= 1e-6|lam| on 80/80 draws),")
print("      so the expansion is about v_b and the law is SECOND order -- which is b61's already-verified")
print("      beat = 1/2 kappa theta^2 (community.md 53.4: per-draw ratio med 1.0000 on the 10 leverage")
print("      draws).  SPACED's slope +1.984 (residual geometric sd 0.091) is that law showing up here.")
print("  (b) BUT A SLOPE IS NOT THE LAW'S EXPONENT -- that is my own rule from 53.3 item 2, which I broke")
print("      while reading DEGEN's +1.539 as a tension with b61.  It is not one: kappa is a per-draw")
print("      quantity through the B2 orientation, it anti-correlates with theta within a cell, and b61's")
print("      own within-cell pooled slope was 1.631, sitting right next to my 1.539.  So the only")
print("      legitimate test of the quadratic law in this design is the per-draw ratio")
print("      beat/(1/2 kappa theta^2) with kappa measured by b61's two independent routes, and I have NOT")
print("      run it.  What D2 does establish rests on a scale-free, per-draw fact instead: v* is NOT a")
print("      critical point of the functional I descend (||grad(v*)|| >= 3.9e8 x its rounding scale, 0/80),")
print("      so theta is the angle between the exact eigenvector block and MY functional's own minimiser --")
print("      a model-discrepancy angle, not an algorithmic-convergence error.")

print("\n" + "=" * 132)
print("IMPLEMENTATION / REGISTRATION HISTORY (post-run additions, disclosed per the stick-change rule)")
print("=" * 132)
print("  1. First run of this file crashed at S0 with `TypeError: '>' not supported between float and list`")
print("     (I wrote max(s0a, max(s0b)) on lists).  Fixed to max(max(s0a), max(s0b)); no sample was consumed")
print("     before the crash, so the 40-draw streams below are unaffected and the numbers are the same")
print("     dataset I would have had.")
print("  2. S3 as PRE-REGISTERED was a Spearman of theta against rho_A WITHIN each fixed spectrum.  I")
print("     registered it without first checking that rho_A can vary there, and it cannot: by S6/A0=MtM")
print("     rho_A = (c1/c2)^2 is fixed by the cosines.  So the printed Spearman values (+0.177, +0.037) are")
print("     correlations against rounding noise and I report them as VOID, not as 'S3 fired -> rho_A is a")
print("     correlate'.  The valid statement from the same data is the INVARIANCE one (theta spreads")
print("     1.87-3.40 decades at rho_A constant).  New discipline: before registering a correlation, print")
print("     the within-group range of the regressor against float noise.")
print("  3. S6 and the DIAGNOSTIC block were added AFTER the first successful run.  S6 rebinds the rng to a")
print("     separate stream (61205) and, in any case, runs after every S1-S5 draw, so it cannot move those")
print("     numbers -- verified by rerunning and diffing S0/S1/S2/S4/S5 (see the txt).  Nothing in")
print("     S6/DIAGNOSTIC gates the write-up; S0/S1/S2/S4/S5 were run exactly as registered.")
print("  4. D2 was added after the DIAGNOSTIC output showed theta and beat correlated within a group")
print("     (Spearman +0.886/+0.888).  It consumes no random numbers (grad_sphere and the extra norms are")
print("     deterministic functions of the already-drawn sample), so S0-S5 are unchanged -- verified by")
print("     diffing the S0-S2 rows of the two runs, which are byte-identical.")
print("  5. A sentence I wrote into the DIAGNOSTIC on the first pass ('the descent reproduces the value at")
print("     v* to ~7e-5 gap in every draw') is FALSE and is withdrawn in the block itself, because the")
print("     per-draw spread printed by that same block is 4.90 (DEGEN) and 3.55 (SPACED) decades; only the")
print("     1.3% agreement of the two MEDIANS survives.  I caught it by reading my own output, not by A.")
print("  6. My first reading of D2 treated DEGEN's pooled slope +1.539 as a tension with b61's quadratic")
print("     law.  That violates my OWN recorded rule (community.md 53.3 item 2: a pooled slope is not the")
print("     law's exponent, because kappa varies per draw and anti-correlates with theta).  Corrected in")
print("     the D2 block before anything reached community.md or the chapter; the per-draw ratio test")
print("     remains un-run and is what b64 should register.")

print("=" * 132)
allr = [r for a in rec.values() for r in a if r is not None]
print(f"  #(testable) = {sum(1 for r in allr if r['test'])}/{len(allr)}   "
      f"#(|grad(v_b)|>1e-6|lam|) = {sum(1 for r in allr if r['gn'] > 1e-6)}")
print(f"  S5 #(beat < 0) = {sum(1 for r in allr if r['beat'] < 0)}   "
      f"#(deficit < 0) = {sum(1 for r in allr if r['defc'] < 0)}   "
      f"min deficit = {min(r['defc'] for r in allr):.4f}")

print("\n" + "=" * 132)
print("判读（预登记，跑前写下的方向；跑后只填数，不改方向）")
print("  S0 先于一切：A0 = MtM 若成立，那么 b63 里我写错的『Λ⁻¹ 偏 weight 小余弦』就有了明确的")
print("     更正对象 —— **A0 完全不含 Λ**，顶谱由**大**余弦决定，小余弦只通过 B2 进来。")
print("     这条恒等式若在数值上站住，它就是 open item 1 里第一个『机制变量』候选的来源。")
print("  S1 是这轮的判决：固定谱、只换子空间，如果 DEGEN 的 θ 中位数仍显著高于 SPACED，")
print("     那么『顶余弦重合控制 θ』就从 b63 的相关升级成了**设计内的对照**（这是我第一次能")
print("     在同一谱下把子空间的贡献分离出来）。若不成立，open item 1 按『我能命名的输入量里")
print("     没有单一控制变量』收尾 —— 那也是一条可以写进正文的负面结果。")
print("  S2 预期**触发**（θ 在 40 个子空间抽样上跨数量级）；它触发得越多，S3 才越有意义：")
print("     ρ_A 若是机制，它必须能解释**同一谱内**的差异，而不只是两谱之间的差异。")
print("  S3 是这次唯一能把 ρ_A 从相关升级成机制的检验，也是唯一可能把我上一轮的推断")
print("     （『顶重合→ρ_A→1→E0 不定→θ 大』）钉死的检验。S3 触发我就写成『ρ_A 只是相关量』。")
