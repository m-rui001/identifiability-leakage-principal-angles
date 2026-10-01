"""
A Agent / D15b = 在 **B 的确切那张表** 上做独立复核。
  B §51.8 的登记数字（来自 b59 的 [D] 段，q=4 单次抽样）：
      λ_min(𝒢)=0.003675032091, cf₁=0.003676813300, λ₁(v*)=0.003675520296（亏 0.274086 of gap）
      B 的 120 条随机射线最好 = 0.003675568071（亏 0.3009 of gap）
  做法：把 B 的 b59 脚本**原样 exec 一遍**（他脚本里 rng 由内部 seed 决定，因此这就是他那一张表），
        取出 (Lam, K, s) 之后，用**我自己的评测器**（numpy.roots 独立求根）与**我自己的优化器**
        （随机球面网格 + 黎曼梯度回溯 + scipy SLSQP）最小化 λ₁(v)。
  纪律：评测器与优化器都是我的；表构造复用他的（这是"同一张表"的唯一办法）。
        地板按 b57/b59 的口径 100·eps·‖G‖₂ 与 gap 一起报（B §54.4 的尺度规矩）。
"""
import numpy as np
import scipy.optimize as spo
import os

os.chdir("../B")
ns = {"__name__": "__main__", "__file__": os.path.abspath("b59_setlevel_gridfree.py")}
src = open("b59_setlevel_gridfree.py", encoding="utf-8").read()
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "b59", "exec"), ns)
Lam, K, s = ns["Lam"], ns["K"], ns["s"]
print("  [已复现 b59 的那张表]  形状 Lam", Lam.shape, " K", K.shape)


def lam_mine(s, v):
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    r = np.roots([C, -(B + C), B * (1.0 - B)])
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[(r > 0) & (r < 1)]
    return (float(r.min()) if r.size else np.nan), B, C


def lam_B(s, v):
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    return float(ns["crossings_stable"](np.array([B]), np.array([C]))[0])


def grad(s, v):
    lam, B, C = lam_mine(s, v)
    disc = (B + C) ** 2 - 4 * C * B * (1 - B)
    Fl = 2 * C * lam - B - C
    lB = -(1 - lam - 2 * B) / Fl
    lC = -(lam * lam - lam) / Fl
    g = 2 * lB * (s["A0"] @ v) + 2 * lC * (s["B2"] @ v)
    return g - v * float(v @ g), lam


def desc(s, v0, iters=6000):
    v = v0 / np.linalg.norm(v0); lam = lam_mine(s, v)[0]; best = (v, lam)
    for _ in range(iters):
        g, _ = grad(s, v); n = np.linalg.norm(g)
        if not np.isfinite(lam) or n == 0:
            break
        improved = False
        for t in (1.0, .3, .1, .03, .01, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10):
            w = v - t * g; nw = np.linalg.norm(w)
            if nw == 0:
                continue
            w = w / nw; lw = lam_mine(s, w)[0]
            if np.isfinite(lw) and lw < lam:
                v, lam, improved = w, lw, True
                if lam < best[1]:
                    best = (v, lam)
                if best[1] < lam + 1e-18:
                    pass
                break
        if not improved:
            break
    g, _ = grad(s, v)
    return best[0], best[1], float(np.linalg.norm(g))


gap = s["cf1"] - s["true"]
Gr = np.block([[s["Lam"], s["K"]], [s["K"].T, np.eye(s["K"].shape[1])]])
floor = 100 * ns["EPS"] * np.linalg.norm(Gr, 2)
l1v = lam_mine(s, s["vstar"])[0]
print("=" * 96)
print(f"  λ_min(𝒢)={s['true']:.12f}   cf₁={s['cf1']:.12f}   gap={gap:.6e}")
print(f"  λ₁(v*)={l1v:.12f}   亏 = {(l1v-s['true'])/gap:.6f} of gap   （B 报 0.274086）")
print(f"  评测器地板 floor=100·eps·‖G‖₂={floor:.3e}   gap/floor={gap/floor:.3e}")
mx = 0.0
rng = np.random.default_rng(12345)
for _ in range(300):
    v = rng.normal(size=s["A0"].shape[0]); v /= np.linalg.norm(v)
    a = lam_mine(s, v)[0]; b = lam_B(s, v)
    if np.isfinite(a) and np.isfinite(b):
        mx = max(mx, abs(a - b))
print(f"  [门] 我的评测器 vs B 的 crossings_stable：最大差 {mx:.3e}")

p = s["A0"].shape[0]
M = 300000
V = rng.normal(size=(M, p)); V /= np.linalg.norm(V, axis=1, keepdims=True)
# 向量化：B(v), C(v)
Bv = np.einsum("ij,jk,ik->i", V, s["A0"], V)
Cv = np.einsum("ij,jk,ik->i", V, s["B2"], V)
disc = np.maximum((Bv + Cv) ** 2 - 4 * Cv * Bv * (1 - Bv), 0.0)
lamV = np.where(Cv > 0, ((Bv + Cv) - np.sqrt(disc)) / (2 * np.where(Cv > 0, Cv, 1.0)), 1.0)
bg = int(np.argmin(lamV))
print(f"  随机球面网格 {M} 点：最好 λ₁={lamV[bg]:.12f}  亏={(lamV[bg]-s['true'])/gap:.6f} of gap")

starts = [s["v1"], s["vstar"], V[bg]] + [V[i] for i in np.argsort(lamV)[:60]]
wA, WA = np.linalg.eigh(s["A0"]); starts.append(WA[:, -1])
wB, WB = np.linalg.eigh(s["B2"]); starts.append(WB[:, -1])
best = (None, np.inf, None)
for st in starts:
    v, lam, gn = desc(s, st)
    if lam < best[1]:
        best = (v, lam, gn)
print(f"  我的黎曼梯度（{len(starts)} 个起点）：最好 λ₁={best[1]:.12f}")
print(f"     亏 = {(best[1]-s['true']):.6e} = {(best[1]-s['true'])/gap:.6f} of gap"
      f"   残梯度={best[2]:.3e}  （B 的 2-平面最好 0.003675568071 = 0.300935 of gap）")
print(f"     是否低于 λ₁(v*)？{best[1] < l1v - 1e-12}   （低于则 B 的 v* 猜想被我这边的搜索击败）")
for meth in ("SLSQP",):
    r = spo.minimize(lambda x: lam_mine(s, x)[0], s["v1"],
                     constraints=[{"type": "eq", "fun": lambda x: x @ x - 1.0}],
                     method=meth, options=dict(maxiter=3000, ftol=1e-18))
    print(f"  scipy {meth}: λ₁={r.fun:.12f}  亏={(r.fun-s['true'])/gap:.6f} of gap")
print()
print("判读：")
print(f" * 我的最好值与亏 = {(best[1]-s['true'])/gap:.6f} of gap；B 的登记门槛是 0.274086。")
print(f" * 亏/地板 = {(best[1]-s['true'])/floor:.3e} ⇒ 亏在地板之上 {np.log10((best[1]-s['true'])/floor):.1f} 个数量级，")
print("   所以'λ₁ 能不能触到 λ_min(𝒢)'在这套算术里是可判定的（不是噪声）。")
