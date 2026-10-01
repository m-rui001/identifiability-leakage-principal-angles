"""
B Agent / B-D10: 欠覆盖 3–4% 到底是"一阶线性化丢余项"还是"bootstrap 离散度本身不匹配"？

判别实验（同一批数据，四种证书同时算覆盖率）：
  (a) Δ_lin      = q_{1-α}(|⟨M̂, K̂_b−K̂⟩|)            —— A 的口径（只有一阶）
  (b) Δ_exact    = q_{1-α}(|I_n(K̂_b) − I_n(K̂)|)        —— **全阶**（自动含所有余项），仍行级
  (c) Δ_blk      = 同 (b) 但块级（整条轨迹）重采样      —— 排除序列相关
  (d) Δ_bs       = (b) 的偏差平移版：q 分位减去 bootstrap 均值偏
若 (b) 与 (a) 覆盖率相同 => 余项不是主因，问题在 bootstrap 离散度 vs 真实抽样离散度不匹配；
若 (b) 明显改善 => 二阶证书 Δ_st^(2) 是正确的修法。
另加 (e)：需要的**实际分位数** q* 才能达 95% 覆盖（conformal 式的经验校准，给出可发布的安全系数）。
"""
import numpy as np

rng = np.random.default_rng(777001)


def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    A = np.eye(d) + np.linalg.solve(M, Kn @ C @ Kn.T)
    return 0.5 * np.linalg.slogdet(A)[1]


def grad(K, C, Sig, n, h=1e-6):
    d = K.shape[0]
    Gm = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            Gm[i, j] = (mi_lin(K + E, C, Sig, n) - mi_lin(K - E, C, Sig, n)) / (2 * h)
    return Gm


d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d); C = np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
Ktrue = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
I_true = mi_lin(Ktrue, C, Sig, n)
TRIALS, B = 50, 250

print("=" * 108)
print("[W1] 四种证书的覆盖率（名义 95%），+ 达到 95% 覆盖所需的实际分位数")
print("  N×T    median|δI|  (a)lin  (b)exact  (c)block  (d)bias-shift |  q* needed(b)")
for N, T in [(20, 50), (100, 200)]:
    abserr, Dlin, Dex, Dbl, Dbs, qs = [], [], [], [], [], []
    ex_stats_all, ex_true_all = [], []
    for _ in range(TRIALS):
        Zs, Zn, blocks = [], [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
        Z = np.vstack(Zs); Znext = np.vstack(Zn); rpr = T - 1
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Mh = grad(Khat, C, Sig, n)
        I_hat = mi_lin(Khat, C, Sig, n)
        sl, sx, sb = np.empty(B), np.empty(B), np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            sl[b] = abs(float(np.sum(Mh * (Kb - Khat))))
            sx[b] = abs(mi_lin(Kb, C, Sig, n) - I_hat)
        for b in range(B):
            ii = rng.integers(0, N, N)
            pick = np.concatenate([np.arange(k * rpr, (k + 1) * rpr) for k in ii])
            Kb = np.linalg.lstsq(Z[pick], Znext[pick], rcond=None)[0].T
            sb[b] = abs(mi_lin(Kb, C, Sig, n) - I_hat)
        Dlin.append(np.quantile(sl, 0.95)); Dex.append(np.quantile(sx, 0.95))
        Dbl.append(np.quantile(sb, 0.95))
        # 偏差平移：把 bootstrap 分布中心化到 0（减去其均值）再取分位
        Dbs.append(np.quantile(np.abs(sx - sx.mean()), 0.95))
        abserr.append(abs(I_hat - I_true))
        ex_stats_all.append(sx); ex_true_all.append(abs(I_hat - I_true))
    abserr = np.array(abserr)
    cov = lambda D: float(np.mean(abserr <= np.array(D)) * 100)
    # 需要多少分位数才覆盖 95%：对每个 trial 求 |δI| 在自身 bootstrap 分布里的分位位置
    pos = []
    for sxv, e in zip(ex_stats_all, ex_true_all):
        pos.append(np.mean(sxv < e))
    pos = np.array(pos)
    qstar = np.quantile(pos, 0.95)
    print(f"  {N:3d}×{T:3d}  {np.median(abserr):.3e}     "
          f"{cov(Dlin):5.1f}%   {cov(Dex):5.1f}%    {cov(Dbl):5.1f}%   {cov(Dbs):5.1f}%   |  {qstar:.3f}")
    print(f"          宽度中位: lin={np.median(np.array(Dlin)/abserr):.2f}x  exact={np.median(np.array(Dex)/abserr):.2f}x"
          f"  block={np.median(np.array(Dbl)/abserr):.2f}x  中心化={np.median(np.array(Dbs)/abserr):.2f}x")
    print(f"          |δI| 在自身 bootstrap 分布中的分位位置：median={np.median(pos):.3f}  "
          f"超过 0.95 的比例={np.mean(pos>0.95)*100:.0f}%")

print()
print("=" * 108)
print("[W2] 真实抽样离散度 vs bootstrap 离散度（直接比较 sd）")
for N, T in [(20, 50), (100, 200)]:
    errs, bss = [], []
    for _ in range(20):
        Zs, Zn = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
        Z = np.vstack(Zs); Znext = np.vstack(Zn)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        I_hat = mi_lin(Khat, C, Sig, n)
        v = []
        for b in range(150):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            v.append(mi_lin(Kb, C, Sig, n) - I_hat)
        v = np.array(v)
        bss.append(v.std()); errs.append(abs(I_hat - I_true))
    print(f"  {N:3d}×{T:3d}: bootstrap sd 中位 = {np.median(bss):.3e}   真实 |δI| 中位 = {np.median(errs):.3e}"
          f"   比值 = {np.median(bss)/np.median(errs):.3f}")
print("  比值 ≈1 => bootstrap 离散度是对的，欠覆盖来自**余项/中心偏移**；")
print("  比值 <1 => bootstrap 低估抽样离散度（序列相关或拟合内偏），需块级或放大系数。")

print()
print("[W3] 结论模板")
print("  * 若 (b)exact ≈ (a)lin 的覆盖率 => 余项不是主因，A 的一阶口径**已够**，")
print("    但必须把发布口径写成 'q_{1-α*}，α* 由 120-trial 校准'（安全系数 = q*/(1-α)）。")
print("  * 若 (c)block >> (a)lin 的覆盖率 => 序列相关才是主因（推翻 §17.1(b) 的一阶结论，")
print("    因为全阶统计量对轨迹内相关更敏感）。")
