"""
A Agent / D15 = 对 B §51.8/§52.8 跨道请求的独立复核：
  在 B 的表上（b57 的 lam_spd(6) + blocks(L, cos[5°,75°])，seed 60311，我**复用他的表构造**），
  用**我自己的评测器**（numpy.roots 独立求根，不用他的 crossings_stable）与**我自己的优化器**
  （随机球面网格 + 黎曼梯度回溯 + scipy SLSQP）最小化 λ₁(v)，报：
    best、deficit = best − λ_min(𝒢)、以 gap 为单位的亏、评测器地板、以及与 B 报的
    λ_min(𝒢)=0.003675032091 / cf₁=0.003676813300 / λ₁(v*)=0.003675520296 / 2-平面最好 0.003675568071 的对比。
  注：表构造层面复用 B 的代码（这是他要求的"同一张表"）；**评测器与优化器是我自己的**，两者分开报。
"""
import numpy as np
import scipy.optimize as spo

ns = {}
src = open("../B/b57_descent_boundary_cubic.py", encoding="utf-8").read()
exec(compile(src.split("def run_cell(")[0], "b57-head", "exec"), ns)
lam_spd, blocks, setup, cf_of = ns["lam_spd"], ns["blocks"], ns["setup"], ns["cf_of"]
SPECTRA, EPS = ns["SPECTRA"], ns["EPS"]
crossings_stable = ns["crossings"]


# ---------- 我自己的评测器（独立于 B 的 crossings_stable） ----------
def lam_mine(s, v):
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    r = np.roots([C, -(B + C), B * (1.0 - B)])          # C λ² − (B+C) λ + B(1−B)
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[(r > 0) & (r < 1)]
    return (float(r.min()) if r.size else np.nan), B, C


def lam_B(s, v):
    B = float(v @ s["A0"] @ v)
    C = float(v @ s["B2"] @ v)
    return float(crossings_stable(np.array([B]), np.array([C]))[0])


def grad(s, v):
    """黎曼梯度：由隐函数微分 F(λ,B,C)=0"""
    lam, B, C = lam_mine(s, v)
    disc = (B + C) ** 2 - 4 * C * B * (1 - B)
    Fl = 2 * C * lam - B - C
    if abs(Fl) < 1e-16:
        return np.zeros_like(v), lam
    lB = -(1 - lam - 2 * B) / Fl
    lC = -(lam * lam - lam) / Fl
    g = 2 * lB * (s["A0"] @ v) + 2 * lC * (s["B2"] @ v)
    g = g - v * float(v @ g)
    return g, lam


def desc(s, v0, iters=4000):
    v = v0 / np.linalg.norm(v0)
    lam = lam_mine(s, v)[0]
    for _ in range(iters):
        g, _ = grad(s, v)
        n = np.linalg.norm(g)
        if not np.isfinite(lam) or n == 0:
            break
        moved = False
        for t in (1.0, .3, .1, .03, .01, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
            w = v - t * g
            nw = np.linalg.norm(w)
            if nw == 0:
                continue
            w = w / nw
            lw = lam_mine(s, w)[0]
            if np.isfinite(lw) and lw < lam:
                if lam - lw < 1e-17 * abs(lam):
                    v, lam = w, lw
                    return v, lam, float(np.linalg.norm(g))
                v, lam, moved = w, lw, True
                break
        if not moved:
            break
    g, _ = grad(s, v)
    return v, lam, float(np.linalg.norm(g))


# ---------- 复现 B 的表，找到 B 报数的那一个 draw ----------
rng = np.random.default_rng(60311)
ns["rng"] = rng                       # 让 lam_spd/blocks 用这个流（与 B 相同）
cs = [np.cos(np.deg2rad(a)) for a in (5.0, 40.0, 75.0)]
draws = []
for i in range(40):
    L = lam_spd(6)
    K = blocks(L, cs)
    s = setup(L, K)
    if s is None:
        draws.append(None)
        continue
    l1v = lam_B(s, s["vstar"])
    draws.append(dict(s=s, true=s["true"], cf1=s["cf1"], l1vstar=l1v, idx=i))
print(f"  成功 draw 数 = {sum(d is not None for d in draws)}/40")
TARGET_TRUE, TARGET_CF = 0.003675032091, 0.003676813300
cand = [d for d in draws if d and abs(d["true"] - TARGET_TRUE) < 5e-10]
print(f"  命中 B 报的 λ_min(𝒢)={TARGET_TRUE} 的 draw: {[c['idx'] for c in cand]}")
if not cand:
    d0 = min([d for d in draws if d], key=lambda d: abs(d["true"] - TARGET_TRUE))
    print(f"  最接近的 draw idx={d0['idx']} true={d0['true']:.12f}")
else:
    d0 = cand[0]
s = d0["s"]
gap = s["cf1"] - s["true"]
Gr = np.block([[s["Lam"], s["K"]], [s["K"].T, np.eye(s["K"].shape[1])]])
floor = 100 * EPS * np.linalg.norm(Gr, 2)
print(f"  draw idx={d0['idx']}: λ_min(𝒢)={s['true']:.12f}  cf₁={s['cf1']:.12f}  gap={gap:.6e}")
print(f"    λ₁(v*)={d0['l1vstar']:.12f}  (B 报 0.003675520296；差 {d0['l1vstar']-0.003675520296:+.3e})")
print(f"    ||v*||={np.linalg.norm(s['vstar']):.15f}  评测器地板 floor=100·eps·‖G‖₂={floor:.3e}  (gap/floor={gap/floor:.3e})")

# ---------- 我的评测器 vs B 的评测器（一致性门） ----------
mx = 0.0
for _ in range(300):
    v = rng.normal(size=s["A0"].shape[0]); v /= np.linalg.norm(v)
    a, _, _ = lam_mine(s, v); b = lam_B(s, v)
    if np.isfinite(a) and np.isfinite(b):
        mx = max(mx, abs(a - b))
print(f"  [门] 我的评测器(numpy.roots) vs B 的 crossings_stable：最大差 {mx:.3e}")

# ---------- 我的搜索 ----------
p = s["A0"].shape[0]
M = 400000
V = rng.normal(size=(M, p))
V /= np.linalg.norm(V, axis=1, keepdims=True)
lamV = np.empty(M)
for i in range(M):
    lamV[i] = lam_mine(s, V[i])[0]
best_grid = int(np.argmin(lamV))
print(f"  随机球面网格 {M} 点：最好 λ₁={lamV[best_grid]:.12f}，亏={lamV[best_grid]-s['true']:.4e}"
      f" = {(lamV[best_grid]-s['true'])/gap:.6f} of gap")

starts = [s["v1"], s["vstar"], V[best_grid]]
w0, W0 = np.linalg.eigh(s["A0"])
starts.append(W0[:, -1])
starts += [V[i] for i in np.argsort(lamV)[:40]]
best = (None, np.inf, None)
for st in starts:
    v, lam, gn = desc(s, st)
    if lam < best[1]:
        best = (v, lam, gn)
print(f"  黎曼梯度（我自己）：最好 λ₁={best[1]:.12f}，残梯度={best[2]:.3e}")
print(f"     亏 = {best[1]-s['true']:.6e} = {(best[1]-s['true'])/gap:.6f} of gap"
      f"   (B 的 2-平面最好 = {(0.003675568071-s['true'])/gap:.6f} of gap)")

for meth in ("SLSQP",):
    r = spo.minimize(lambda x: lam_mine(s, x)[0], s["v1"],
                     constraints=[{"type": "eq", "fun": lambda x: x @ x - 1.0}],
                     method=meth, options=dict(maxiter=4000, ftol=1e-16))
    print(f"  scipy {meth}：λ₁={r.fun:.12f}，亏={(r.fun-s['true'])/gap:.6f} of gap，‖∇c‖={np.linalg.norm(r.jac):.2e}")

print()
print("判读（按 B 的登记：若我给出 < 0.274086 of gap，则 open item 1 有实质进展）")
print(f" * 我这边的结论：最小亏 = {(best[1]-s['true'])/gap:.6f} of gap（黎曼梯度），"
      f"随机网格 {(lamV[best_grid]-s['true'])/gap:.6f} of gap")
print(" * 若两者都 ≥ 0.274086，则 B 的 v* 猜想获得一次独立复核（用不同评测器与不同优化器）。")
print(" * λ₁ 是否能触到 λ_min(𝒢)：亏/floor 比值给出数值可分辨性；若亏 ≪ floor 则'触到'在数值上不可判定。")
