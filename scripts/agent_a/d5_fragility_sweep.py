"""
A Agent / D5: A-CLAIM-7 —— 相对信息脆弱性是否随谱崩塌单调上升？
  fragility := |I_n(K+E) - I_n(K)| / I_n(K),   固定 ||E||_F, 固定 n
  谱崩塌度用两个基无关指标度量：
    (i) 参与率 PR = (sum mu^2)^2 / sum mu^4  （有效模态数）
    (ii) 保留维度 d_eff（在给定 IB 预算 tau 下 water-filling 保留的模态数）
  谱随机化：mu_j = exp(-gamma_j), gamma ~ 混合分布（让 PR 有跨度）
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr

rng = np.random.default_rng(2026)
d, sigma2 = 8, 1e-3
Sig = sigma2 * np.eye(d)


def MI(K, C, Sig, n):
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    P = Kn @ C @ Kn.T
    A = np.eye(K.shape[0]) + np.linalg.solve(M, P)
    return 0.5 * np.linalg.slogdet(A)[1]


def m_of_gamma(gamma, n, sigma2):
    gamma = np.asarray(gamma, float)
    return np.where(gamma < 1e-12, n * sigma2,
                    sigma2 * (1 - np.exp(-2 * n * gamma)) / (1 - np.exp(-2 * gamma)))


def optimal_c(mu, n, sigma2, tau):
    """注意：入参是 mu（不是 gamma）——v1 版本此处形参命名错误导致 d_eff 全为 1（已修正）"""
    mu = np.asarray(mu, float)
    gamma = -np.log(np.abs(mu))
    inv_a = m_of_gamma(gamma, n, sigma2) / np.abs(mu) ** (2 * n)
    order = np.argsort(inv_a); s = np.sort(inv_a); c = np.zeros_like(s); th = None
    for k in range(1, len(s) + 1):
        cand = (tau + s[:k].sum()) / k
        if (k == len(s) or cand <= s[k]) and cand > s[k - 1] - 1e-15:
            th = cand; c[:k] = cand - s[:k]; break
    out = np.empty_like(c); out[order] = c
    return out, th


n, tau, eps = 20, 1.0, 1e-3
rows = []
for trial in range(400):
    # 随机谱：对数均匀 + 随机"崩塌"程度
    k = rng.integers(1, d + 1)
    g1 = rng.uniform(1e-4, 5e-3)          # 慢带
    g2 = rng.uniform(2e-2, 3e-1)          # 快带
    gamma = np.concatenate([rng.uniform(g1, g1 * 3, k), rng.uniform(g2, g2 * 2, d - k)])
    mu = np.exp(-gamma)
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    K = Q @ np.diag(mu) @ Q.T
    C = np.eye(d)
    E = rng.normal(size=(d, d)); E = E / np.linalg.norm(E, "fro") * eps
    I0 = MI(K, C, Sig, n)
    dI = abs(MI(K + E, C, Sig, n) - I0)
    frag = dI / I0
    pr = (np.sum(mu ** 2) ** 2) / np.sum(mu ** 4)
    c_, th = optimal_c(np.sort(mu), n, sigma2, tau)
    deff = int((c_ > 1e-14).sum())
    rows.append((pr, deff, frag, I0, dI))

import numpy as np
rows = np.array(rows)
pr, deff, frag, I0, dI = rows.T
print("=" * 78)
print("[D5] 400 组随机谱 (d=8, n=20, ||E||=1e-3, tau=1) 的相对脆弱性 frag=|dI|/I")
print(f"  frag:  min={frag.min():.3e}  median={np.median(frag):.3e}  max={frag.max():.3e}  跨度={frag.max()/frag.min():.1f}x")
print()
print("  相关（frag 对 谱崩塌度）：")
print(f"    Spearman(frag, 参与率 PR)      = {spearmanr(frag, pr)[0]:+.3f}  p={spearmanr(frag, pr)[1]:.2e}")
print(f"    Spearman(frag, d_eff@tau=1)    = {spearmanr(frag, deff)[0]:+.3f}  p={spearmanr(frag, deff)[1]:.2e}")
print(f"    Spearman(frag, I_n)            = {spearmanr(frag, I0)[0]:+.3f}  p={spearmanr(frag, I0)[1]:.2e}")
print()
print("  按 d_eff 分组的 frag 中位数：")
for k in sorted(set(deff.astype(int))):
    m = deff == k
    print(f"    d_eff={k}: n={m.sum():4d}   median frag={np.median(frag[m]):.3e}   "
          f"IQR=[{np.percentile(frag[m],25):.2e}, {np.percentile(frag[m],75):.2e}]")
print()
print("  按 PR 四分位分组：")
qs = np.quantile(pr, [0, .25, .5, .75, 1.0])
for i in range(4):
    m = (pr >= qs[i]) & (pr <= qs[i + 1])
    print(f"    PR∈[{qs[i]:.2f},{qs[i+1]:.2f}]: median frag={np.median(frag[m]):.3e}  I_n 中位数={np.median(I0[m]):.3f}")
print()
print("  判定：若 Spearman(frag, PR) 显著为负且跨度 ≥2x => A-CLAIM-7 成立；否则否证。")
