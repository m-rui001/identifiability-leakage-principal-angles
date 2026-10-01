"""
b44 -- WHICH hypothesis actually costs cf its exactness: the angle SPREAD, or the anisotropy of Lam?

Trigger (self-correction candidate #18).  b42 [T3] reported, at p=4/q=3 with a PRESCRIBED
principal-cosine spectrum, that the scalar closed form cf(t,alpha) of prop:plane
  * stays within <=3.4% of lambda_min(Gr) in SIZE for every spectrum, but
  * changes SIGN (cf < lambda_min, 119-120/120 draws) in the two cells whose cosine spectrum is
    spread.
I read that as "the angle distribution moves lambda_min by <=3.4% and flips cf's side", and wrote
it into chapter4 (rem:multi (ii)) and into community 39.3 as a reason to RETIRE the angle-distribution
direction.  While writing 40.7 I checked the isotropic case by hand and it does not fit that reading:

  LAM = I_p  =>  K = M, sigma_max(K) = c_1, and Gr = [[I,K],[K^T,I]] has lambda_min = 1 - c_1
  (the singular values of K come in blocks 1 +- sigma_i, plus 1's on the p-q leftover directions).
  On the other hand cf's inputs at Lam = I are
      c2 = eigvalsh(K^T K)_max = c_1^2,   t = ||K||_2^2 / c_1^2 = 1,   alpha_top = arccos(c_1),
      cf(1, alpha) = (2 - sqrt(4 - 4 sin^2 alpha))/2 = 1 - cos alpha = 1 - c_1.
  So cf is EXACTLY lambda_min at Lam = I, for EVERY cosine spectrum -- equal or spread, any p >= q.

If that is right, then the 1e-3-level spread-cell gaps of b42 CANNOT be caused by the spread alone:
they need a non-isotropic Lam.  The angle distribution would then not be the variable at all, and
39.3's "the angle distribution is a <=3.4% effect" would be a MISATTRIBUTION (the effect is real, the
label on it is wrong).  This script separates the two factors with a 2 x 5 design and adds the
resolution audit that 40.7 asked for (absolute gaps against the eigvalsh floor, so the SIGN flip can
be declared noise or algebra).

Pre-registered, before looking at any output:
  T1 (isotropy column).  At Lam = I_p every spectrum cell, spread included, must have
      max|cf - true| at the round-off floor (< 1e-14, since the analytic gap is identically 0).
      Falsifier: a spread cell at Lam = I with |cf - true| >> 1e-14.  That would mean my hand
      derivation above is wrong, and cf is NOT exact at isotropy -- then 40/39 need no correction,
      but prop:plane loses a case I believed it had (cor:square's p = q exactness would be the only
      exactness locus left).
  T2 (attribution).  At general Lam (unit-norm Gram, kappa ~ 1.6) the spread rows must reproduce
      b42's 1e-3..3e-2 gaps and the sign flip; the equal-angle rows must stay small.  If the flip
      survives at general Lam but the entire isotropic column is at the floor, the variable is
      ANISOTROPY, not angle spread, and rem:multi (ii) + 39.3 must be re-labelled.
  T3 (resolution audit, the 40.7 item).  Per cell report min lambda_min, the absolute gaps, and the
      per-draw floor 100*eps*||Gr||_2; a draw is 'unresolved' if |cf - true| is below its floor.
      Falsifier: if the violating draws are unresolved (gaps at the floor), the sign flip is an
      eigvalsh artifact and rem:multi (ii)'s "cf stops being an upper estimate" must be retracted.
  T4 (dose -- optional, only if T2 lands as predicted).  Lam = diag(1,1,1,1-delta) with the weak
      direction ALIGNED with or TRANSVERSE to K's top right-singular direction, delta = 1e-3, 1e-6.
      If the gap grows with delta only in one alignment, the controlling quantity is the anisotropy
      IN the range of K, not ||Lam|| spread in general -- which would be the first statement about
      lambda_min that is not a function of t alone.
"""
import numpy as np

rng = np.random.default_rng(46211)
D = 100
P, QN = 4, 3
EPS = np.finfo(float).eps

SPECTRA = {
    "equal 20d":       [np.cos(np.deg2rad(20))] * 3,
    "equal 5d":        [np.cos(np.deg2rad(5))] * 3,
    "spread 5-75":     [np.cos(np.deg2rad(a)) for a in (5, 40, 75)],
    "spread 2-45-88":  [np.cos(np.deg2rad(a)) for a in (2, 45, 88)],
    "spread 30-88-89": [np.cos(np.deg2rad(a)) for a in (30, 88, 89)],
}


def lam_spd(p):
    """Gram of p unit-norm dictionary vectors: general, well-conditioned, diag = 1 (b42, verbatim)."""
    E = rng.normal(size=(D, p))
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E.T @ E


def blocks(Lam, cosines):
    """Prescribed principal-cosine spectrum: K = Lam^{1/2} Q1 diag(c) Q2^T (b42, verbatim)."""
    p, q = Lam.shape[0], len(cosines)
    Q1, _ = np.linalg.qr(rng.normal(size=(p, q)))
    Q2, _ = np.linalg.qr(rng.normal(size=(q, q)))
    M = Q1 @ np.diag(cosines) @ Q2.T
    ev, V = np.linalg.eigh(Lam)
    Lam_half = V @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ V.T
    K = Lam_half @ M
    return K, M


def cf_scalar(Lam, K, alpha_deg):
    """prop:plane closed form as a function of the scalar t only (b42, verbatim)."""
    c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))[-1]
    t = np.linalg.norm(np.linalg.solve(Lam, K), 2) ** 2 / c2
    s = np.sin(np.deg2rad(alpha_deg)) ** 2
    disc = max((1.0 + t) ** 2 - 4.0 * t * s, 0.0)
    return ((1.0 + t) - np.sqrt(disc)) / (2.0 * t) if t > 0 else s


def cell(Lam_of, cs, n=100):
    """Return (gaps, lam_mins, floors, top-cosine alphas) over n draws that pass true > 1e-12."""
    gaps, lm, fl = [], [], []
    skipped = 0
    for _ in range(n):
        Lam = Lam_of()
        K, _ = blocks(Lam, cs)
        Gr = np.block([[Lam, K], [K.T, np.eye(len(cs))]])
        true = np.linalg.eigvalsh(Gr)[0]
        if true <= 1e-12:
            skipped += 1
            continue
        a_top = np.degrees(np.arccos(min(1.0, max(cs))))
        gaps.append(cf_scalar(Lam, K, a_top) - true)
        lm.append(true)
        fl.append(100.0 * EPS * np.linalg.norm(Gr, 2))
    return np.array(gaps), np.array(lm), np.array(fl), skipped


def report(tag, gaps, lm, fl, skipped):
    if gaps.size == 0:
        print(f"   {tag:>26}   (no usable draw)")
        return
    abs_g = np.abs(gaps)
    rel_g = abs_g / lm
    unres = int((abs_g < fl).sum())
    res = abs_g >= fl
    print(f"   {tag:>26}  n={gaps.size:3d} skip={skipped:2d}  min lam_min={lm.min():9.2e}"
          f"  med|rel gap|={np.median(rel_g):9.2e}  max|gap|={abs_g.max():9.2e}"
          f"  min|gap|={abs_g.min():9.2e}")
    print(f"   {'':>26}  max floor={fl.max():9.2e}  unresolved(|gap|<floor)={unres:3d}"
          f"  cf<true: all={int((gaps<0).sum()):3d} / resolved-only={int((gaps[res]<0).sum()):3d}"
          f"  min|gap|/max-floor={abs_g.min() / fl.max():11.3e}")


print("=" * 118)
print("[T1+T2] 2 x 5 design: Lam isotropic (I_4) vs general (unit-norm Gram); 5 cosine spectra;")
print("        100 draws/cell, p=4, q=3.  cf uses the MOST GENEROUS alpha: the top cosine.")
print()
print("  Lam = I_p   (predicted: gap identically 0 in EVERY cell, spread included)")
for name, cs in SPECTRA.items():
    g, m, f, sk = cell(lambda: np.eye(P), cs)
    report(name, g, m, f, sk)
print()
print("  Lam = general spd (predicted: spread rows ~1e-3..3e-2 and cf<true; equal rows small)")
for name, cs in SPECTRA.items():
    g, m, f, sk = cell(lambda: lam_spd(P), cs)
    report(name, g, m, f, sk)

print()
print("[T4] dose of anisotropy.  Lam = diag(1,1,1,1-delta), delta=1e-3 / 1e-6, with the weak")
print("     direction ALIGNED with K's top right-singular direction vs TRANSVERSE to it.")
print("     Predicted ONLY IF T2 lands: aligned >> transverse, and growing with delta.")
for cs_name in ["equal 5d", "spread 2-45-88"]:
    for delta in [1e-3, 1e-6]:
        for align in ["aligned", "transverse"]:
            def lam_of(delta=delta, align=align):
                Lam = np.diag([1.0, 1.0, 1.0, 1.0 - delta])
                if align == "transverse":
                    # rotate the weak direction off the last coordinate
                    th = np.deg2rad(45.0)
                    R = np.eye(P)
                    R[0, 0], R[0, 3], R[3, 0], R[3, 3] = np.cos(th), -np.sin(th), np.sin(th), np.cos(th)
                    return R @ Lam @ R.T
                return Lam
            g, m, f, sk = cell(lam_of, SPECTRA[cs_name])
            report(f"{cs_name} {align} d={delta:.0e}", g, m, f, sk)
    print()

print("判读（**预登记**判据，跑之前写的；实测结果在文件末尾 '实测判读'）：")
print("  T1/T2 若 isotropic 列全部 |gap| ~ 1e-16 而 general 列的 spread 行到 1e-3 ——")
print("     则 'spread 造成 3.4% 与符号翻转' 是**误标**：变量是 Λ 的各向异性，角度分布本身不动 cf。")
print("  T1 若在 isotropic 列出现任何一个 |gap| > 1e-14 —— 我上面那段手推（cf(1,alpha_top)=1-c_1=lambda_min）")
print("     就是错的，必须先收回它，再谈 b42 的归因。")
print("  T3 若 violation 全落在 unresolved 里 —— b42 的符号翻转是 eigvalsh 地板，rem:multi (ii) 撤回。")
print()
print("[T5] is the exactness locus 'Lambda = I_p' or the weaker 'Lambda = gamma * I_p'?")
print("     Lambda = gamma*I_p, spread 2-45-88 and equal 20d, gamma = 1 / 1.5 / 0.6.")
print("     Under Lambda = I the truth is analytic (1 - c_1); for gamma != 1 the block")
print("     [[gamma, s],[s, 1]] gives lambda = ((gamma+1) - sqrt((gamma+1)^2 - 4(gamma - s^2)))/2")
print("     with s = gamma^{1/2} c_1, so a genuine closed form exists -- compare it to cf(t, alpha_top),")
print("     where t = ||Lam^{-1}K||_2^2 / c_1^2 = 1/gamma here.  Prediction: cf is NOT exact for")
print("     gamma != 1 -- the locus is isotropy AND the unit-norm normalisation together (Lam = I_p),")
print("     because cf has no knowledge of gamma.")
for cs_name in ["spread 2-45-88", "equal 20d"]:
    for gamma in [1.0, 1.5, 0.6]:
        gaps, lm, fl, sig = [], [], [], []
        for _ in range(100):
            Lam = gamma * np.eye(P)
            cs = SPECTRA[cs_name]
            K, _ = blocks(Lam, cs)
            Gr = np.block([[Lam, K], [K.T, np.eye(len(cs))]])
            true = np.linalg.eigvalsh(Gr)[0]
            if true <= 1e-12:
                continue
            a_top = np.degrees(np.arccos(min(1.0, max(cs))))
            gaps.append(cf_scalar(Lam, K, a_top) - true)
            lm.append(true)
            fl.append(100.0 * EPS * np.linalg.norm(Gr, 2))
            sig.append(np.linalg.norm(np.linalg.solve(Lam, K), 2) ** 2
                       / np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))[-1])
        g, m, f = np.array(gaps), np.array(lm), np.array(fl)
        ag = np.abs(g)
        print(f"   {'gamma=' + format(gamma, '.1f') + ' ' + cs_name:>0}: n={g.size:3d}"
              f"  med|rel gap|={np.median(ag / m):9.2e}  max|gap|={ag.max():9.2e}"
              f"  min|gap|/floor={(ag.min() / f.max()):9.2e}  mean t={np.mean(sig):7.4f}"
              f"  cf<true={int((g < 0).sum()):3d}")

print()
print("[T6] the T5 reading must hold for an ARBITRARY block K, not only for my prescribed-spectrum")
print("     construction.  Claim under test: if Lam = gamma*I_p then, with NO hypothesis on the")
print("     singular values of K (equal angles / spread / rank-deficient) and no hypothesis on which")
print("     of p, q is larger,")
print("        lambda_min(Gr) = ((gamma+1) - sqrt((gamma-1)^2 + 4*gamma*c^2))/2,")
print("        c^2 = lam_max(K^T Lam^{-1} K),   i.e. c = sigma_max(K)/gamma^{1/2}, the top cosine.")
print("     Proof: K = U diag(sigma) V^T is an orthogonal change of variables on Gr, which reduces to")
print("     blocks [[gamma, sigma_i],[sigma_i, 1]] (minus branch above at c = sigma_i/gamma^{1/2}),")
print("     plus gamma on ker K^T and 1 on ker K; the minus branch is <= gamma for every gamma > 0")
print("     (square both sides: (1-gamma)^2 <= (1-gamma)^2 + 4 gamma c^2), so the overall minimum is")
print("     the sigma_max branch.  cf has t = ||Lam^{-1}K||_2^2/c^2 = 1/gamma in this case, and")
print("     cf(1/gamma, alpha) expands to the same branch, so this is cf being EXACT, not a new formula.")
for (pp, qq) in [(4, 3), (3, 3), (3, 5)]:
    for ctop in [0.3, 0.9, 0.999]:
        for rank in [min(pp, qq), 1]:
            e_cf, e_cf_g = [], []
            for _ in range(40):
                G = rng.normal(size=(pp, qq))
                if rank < min(pp, qq):
                    G[:, rank:] = 0.0
                s = np.linalg.svd(G, compute_uv=False)
                for gamma in [1.0, 1.5, 0.6]:
                    # scale so that the TOP COSINE c = sigma_max(K)/gamma^{1/2} is ctop, which keeps
                    # K^T Lam^{-1} K <= I (otherwise Gr is not psd and the chapter is not in range).
                    K = G * (np.sqrt(gamma) * ctop / s[0])
                    Lam = gamma * np.eye(pp)
                    Gr = np.block([[Lam, K], [K.T, np.eye(qq)]])
                    true = np.linalg.eigvalsh(Gr)[0]
                    c2 = np.linalg.eigvalsh(K.T @ np.linalg.solve(Lam, K))[-1]
                    pred = ((gamma + 1.0) - np.sqrt((gamma - 1.0) ** 2 + 4.0 * gamma * c2)) / 2.0
                    pred_cf = cf_scalar(Lam, K, np.degrees(np.arccos(np.sqrt(min(c2, 1.0)))))
                    e_cf.append(abs(pred - true))
                    e_cf_g.append(abs(pred_cf - true))
            print(f"   p={pp} q={qq} top cosine={ctop:5.3f} rank={rank:2d}"
                  f"  max|branch-eigvalsh|={max(e_cf):9.2e}  max|cf-eigvalsh|={max(e_cf_g):9.2e}")

print()
print("实测判读（本节数字来自本文件同一次运行，seed 46211，可逐位复现）：")
print("  T1 isotropic 列：五个 spectrum 全部 max|gap| <= 1.33e-15，且 100/100 落在各自 floor 之下")
print("     （cf<true 50/35/47 这类数字是**零间隔的硬币**，见 T1 控制组的意义）。手推成立：")
print("     Λ 标量化之后 cf 与 λ_min(𝒢) 恒等，与角度分布**无关**。")
print("  T2 general Λ：equal 行 med 相对 gap 1.3e-4(20d)/8.8e-6(5d)，spread 行 1.4e-3..2.7e-3；")
print("     符号翻转只在 5-75（100/100 resolved）与 2-45-88（99/100，resolved 99）出现，")
print("     30-88-89 **不翻**（0/100）。=> 变量是 Λ 的各向异性 × '第二个不小的余弦'，")
print("     不是'角度是否分散'本身；b42/§39 把它写成 angle-distribution effect 是**误标**。")
print("  T3 分辨率审计：这两个翻转格的 min|gap|/max-floor = 9.1e5 与 1.1e4，unresolved = 0")
print("     => 翻转是代数事实，不是 eigvalsh 地板。rem:multi (ii) 站得住，但归因要改。")
print("  T4 **设计失败，如实记录**：'aligned / transverse' 没有控制任何东西 —— blocks() 里的 Q1 是")
print("     独立随机 QR，弱方向 e_4 并不与 K 的 top right-singular direction 对齐；我把 Λ 旋转 45°")
print("     之后，随机 Q1 把旋转吸收掉了，两格其实是同一个 ensemble（数字 6.66e-9 vs 7.11e-9）")
print("     这正是'控制无效'的签名。alignment 这个问题**没有被检验**，仍属 open。")
print("  T5 预登记说'γ≠1 时 cf 不再精确，因为 cf 不知道 γ' —— **被自己的数据推翻**：γ=1.5/0.6 的")
print("     gap 全在 1e-15 地板，t 实测恰为 1/γ（0.6667/1.6667）。手推之后明白为什么：")
print("     cf(1/γ, α) 展开恰好等于 [[γ,σ],[σ,1]] 的 minus 分支。所以精确性 locus 是")
print("     **Λ 标量（任意 γ）**，不是 'Λ=I 且 diag=1'。")
print("  T6 任意 K（含 q>p、rank=1、top cosine 0.3/0.9/0.999）× γ∈{1,1.5,0.6}，18 格：分支式与 cf 都到")
print("     eigvalsh 的 1.7e-15 级 => 上面那段 SVD 约化证明成立，且不依赖我的 prescribe-spectrum 构造。")
print("  证书 (★) 不受影响：cf 走上界侧。本节的后果全在'如何描述 cf 的精确性 locus'。")
