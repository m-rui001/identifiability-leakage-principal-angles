"""
B Agent / B-D9: 把 §17.1 的三个悬而未决一次做完（trials=120，而不是 A 的 40）

对照三种证书的**覆盖率**与**宽度**（同一批数据，三种口径都算）：
  (1) Δ_wc      = ‖M̂‖_F · ρ̂          —— 修正后的最坏情况（A 的 d9 用的是谱范数，不合法，见 §17.1(a)）
  (2) Δ_st      = q_0.95 |⟨M̂, K̂_b − K̂⟩| —— A 的统计口径（一阶 + 行级 bootstrap）
  (3) I_boot    = [MI(K̂)+q_0.05, MI(K̂)+q_0.95] 中心 bootstrap **区间**（含非线性与偏差）
                    以及偏差校正版 I_bc = [MI(K̂)−b̂+q_0.05, ...]，b̂ = mean_b MI(K̂_b) − MI(K̂)

判据（替 A 把 K1 写清楚）：
  * 一维"上界"证书要覆盖 |δI|（单侧）；"区间"证书要覆盖 δI（双侧）。两者覆盖率不可混为一谈。
  * 若 I_boot 覆盖 ≈90-95% 而 Δ_st 覆盖 <90%，说明欠覆盖来自**一阶线性化丢掉的偏差**，
    不是 resampling 方案（§17.1(b) 已排除 resampling）。
"""
import numpy as np

rng = np.random.default_rng(20261001)


def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    A = np.eye(d) + np.linalg.solve(M, Kn @ C @ Kn.T)
    return 0.5 * np.linalg.slogdet(A)[1]


def deriv_matrix(K, C, Sig, n, h=1e-6):
    d = K.shape[0]
    Gm = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = h
            Gm[i, j] = (mi_lin(K + E, C, Sig, n) - mi_lin(K - E, C, Sig, n)) / (2 * h)
    return Gm


d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d)
C = np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
Ktrue = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
I_true = mi_lin(Ktrue, C, Sig, n)
TRIALS, B = 120, 200

print("=" * 104)
print("[Z1] 三种证书的覆盖率与宽度（trials=%d, B=%d；δI = MI(K̂)-MI(K*) 的真值）" % (TRIALS, B))
print("  N×T   median|δI|  Δ_wc/|δI| 覆盖(1侧) | Δ_st/|δI| 覆盖(1侧) | 中心区间覆盖(名义90%) | 偏差平移区间覆盖")
for N, T in [(20, 50), (50, 100), (100, 200)]:
    rows = []
    for _ in range(TRIALS):
        Zs, Zn = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
        Z = np.vstack(Zs); Znext = np.vstack(Zn)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        Mh = deriv_matrix(Khat, C, Sig, n)
        idx = rng.permutation(len(Z)); hh = len(idx) // 2
        K1 = np.linalg.lstsq(Z[idx[:hh]], Znext[idx[:hh]], rcond=None)[0].T
        K2 = np.linalg.lstsq(Z[idx[hh:]], Znext[idx[hh:]], rcond=None)[0].T
        rho = np.linalg.norm(K1 - K2, "fro") / np.sqrt(2)
        D_wc = np.linalg.norm(Mh, "fro") * rho          # 修正：Frobenius
        st = np.empty(B); mi_b = np.empty(B)
        for b in range(B):
            ib = rng.integers(0, len(Z), len(Z))
            Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
            st[b] = abs(float(np.sum(Mh * (Kb - Khat))))
            mi_b[b] = mi_lin(Kb, C, Sig, n)
        D_st = np.quantile(st, 0.95)
        I_hat = mi_lin(Khat, C, Sig, n)
        # bootstrap 分布近似 (MI(K̂)-MI(K*)) 的抽样分布 =>
        #   中心百分位 CI for I_true = [I_hat - q95, I_hat - q05]
        #   偏差平移 CI            = [I_hat - bhat + q05, I_hat - bhat + q95]  其中 bhat=mean_b(MI_b)-I_hat
        qlo, qhi = np.quantile(mi_b - I_hat, 0.05), np.quantile(mi_b - I_hat, 0.95)
        bhat = mi_b.mean() - I_hat
        lo, hi = I_hat - qhi, I_hat - qlo
        clo, chi = I_hat - bhat + qlo, I_hat - bhat + qhi
        rows.append((abs(I_hat - I_true), I_hat - I_true, D_wc, D_st, lo, hi, clo, chi,
                     np.linalg.norm(Mh, 2) * rho, I_true))
    a = np.array(rows)
    dI, Dwc, Dst = a[:, 1], a[:, 2], a[:, 3]
    cov_wc1 = np.mean(np.abs(dI) <= Dwc)
    cov_st1 = np.mean(np.abs(dI) <= Dst)
    cov_c2 = np.mean((a[:, 4] <= a[:, 9]) & (a[:, 9] <= a[:, 5]))
    cov_bc2 = np.mean((a[:, 6] <= a[:, 9]) & (a[:, 9] <= a[:, 7]))
    # 二侧证书（上界也当区间用）： |dI|<=D 等价 [−D, +D] 覆盖
    cov_wc2 = cov_wc1; cov_st2 = cov_st1
    print(f"  {N:3d}×{T:3d}  {np.median(a[:,0]):.3e}   {np.median(Dwc/a[:,0]):7.1f}  {cov_wc1*100:5.1f}%  "
          f"| {np.median(Dst/a[:,0]):7.2f}  {cov_st1*100:5.1f}%  | {cov_c2*100:5.1f}%            | {cov_bc2*100:5.1f}%")
    print(f"            （A 原口径 ‖M‖₂ρ 的覆盖率 = {np.mean(np.abs(dI) <= a[:,8])*100:.1f}%  "
          f"vs 修正后 ‖M‖_Fρ = {cov_wc1*100:.1f}%；宽度中位比 = {np.median(a[:,2]/a[:,8]):.3f}）")

print()
print("=" * 104)
print("[Z2] 偏差的来源分解：|δI| 里有多少是**系统性偏差**（bootstrap 抓不到的部分）")
for N, T in [(50, 100)]:
    dIs = []
    for _ in range(TRIALS):
        Zs, Zn = [], []
        for _ in range(N):
            z = rng.normal(size=d); traj = [z]
            for _ in range(T - 1):
                z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
            X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
        Z = np.vstack(Zs); Znext = np.vstack(Zn)
        Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
        dIs.append(mi_lin(Khat, C, Sig, n) - I_true)
    dIs = np.array(dIs)
    print(f"  N={N},T={T}: mean δI = {dIs.mean():+.4e} (sd {dIs.std()/np.sqrt(TRIALS):.2e})   "
          f"median|δI| = {np.median(np.abs(dIs)):.4e}")
    print(f"  ⇒ 系统性偏差占典型误差的比例 = {abs(dIs.mean())/np.median(np.abs(dIs))*100:.1f}%")
    print(f"     若用**单侧上界**证书（|δI|≤Δ），偏差贡献 {(np.mean(dIs>0)) * 100:.0f}% 的样本落在正侧 "
          f"=> 上界口径在符号不对称时天然偏保守或偏激进，需在论文里声明是单侧还是双侧。")

print()
print("[Z3] 判读规则（写给下一轮的决策）")
print("  * 若 Δ_st 单侧覆盖 ≥90% 且 Δ_st/|δI| ≤ 5：K1 通过（一阶统计证书可用）。")
print("  * 若 中心区间覆盖明显 <95%：需要二阶修正或 log-det 的解析偏差项（第 3 章仍有工作）。")
print("  * 若 Δ_wc(Frobenius) 覆盖 =100% 且宽度比 ~10-20×：保留为'认证口径'，与统计口径并列发布。")
