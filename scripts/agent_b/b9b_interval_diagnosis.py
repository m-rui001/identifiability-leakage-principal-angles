"""
B Agent / B-D9b: 诊断 [Z1] 里"bootstrap 区间覆盖 = 0%"是真实现象还是我的 bug
输出 bootstrap 分布的位置/形状 与 真值 δI 的分布，直接看区间为什么错侧。
"""
import numpy as np

rng = np.random.default_rng(555)


def mi_lin(K, C, Sig, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K, i) @ Sig @ np.linalg.matrix_power(K, i).T for i in range(n))
    A = np.eye(d) + np.linalg.solve(M, Kn @ C @ Kn.T)
    return 0.5 * np.linalg.slogdet(A)[1]


d, n, sigma2 = 6, 10, 1e-3
Sig = sigma2 * np.eye(d); C = np.eye(d)
Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
Ktrue = Q @ np.diag([0.999, 0.99, 0.95, 0.85, 0.6, 0.3]) @ Q.T
I_true = mi_lin(Ktrue, C, Sig, n)
N, T, B, TRIALS = 50, 100, 300, 12


def data():
    Zs, Zn = [], []
    for _ in range(N):
        z = rng.normal(size=d); traj = [z]
        for _ in range(T - 1):
            z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
        X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
    return np.vstack(Zs), np.vstack(Zn)


print("=" * 100)
print("[D1] 每个 trial: δI(真) 与 bootstrap 分布 (mi_b - I_hat) 的分位数")
print("  trial   I_hat-I_true    q05      q25     median    q75     q95     P(mi_b<I_hat)")
for t in range(TRIALS):
    Z, Znext = data()
    Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
    I_hat = mi_lin(Khat, C, Sig, n)
    mi_b = np.empty(B)
    for b in range(B):
        ib = rng.integers(0, len(Z), len(Z))
        Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
        mi_b[b] = mi_lin(Kb, C, Sig, n)
    dl = mi_b - I_hat
    qs = np.quantile(dl, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"   {t:3d}   {I_hat-I_true:+.4e}   " + "  ".join(f"{q:+.3e}" for q in qs)
          + f"    {np.mean(dl < 0):.2f}")

print()
print("=" * 100)
print("[D2] 为什么错侧？检查 MI 是否被最小二乘拟合'最大化'：比较 ||K̂_b − K̂|| 与 ||K̂ − K*||")
Z, Znext = data()
Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
print(f"  ||K̂−K*||_F = {np.linalg.norm(Khat-Ktrue,'fro'):.4e}")
nb = []
for b in range(100):
    ib = rng.integers(0, len(Z), len(Z))
    Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
    nb.append(np.linalg.norm(Kb - Khat, "fro"))
nb = np.array(nb)
print(f"  bootstrap 分散 ||K̂_b−K̂||_F: median={np.median(nb):.4e}  q95={np.quantile(nb,0.95):.4e}")
print(f"  比值 median/||K̂−K*|| = {np.median(nb)/np.linalg.norm(Khat-Ktrue,'fro'):.3f}  (应≈1 才说明离散度对)")

print()
print("[D3] MI 的**一阶**离散度 vs bootstrap 的 MI 离散度（判断非线性/符号偏）")
h = 1e-6
Gm = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        E = np.zeros((d, d)); E[i, j] = h
        Gm[i, j] = (mi_lin(Khat + E, C, Sig, n) - mi_lin(Khat - E, C, Sig, n)) / (2 * h)
lin_sd = []
mi_b = []
for b in range(200):
    ib = rng.integers(0, len(Z), len(Z))
    Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
    lin_sd.append(float(np.sum(Gm * (Kb - Khat))))
    mi_b.append(mi_lin(Kb, C, Sig, n) - mi_lin(Khat, C, Sig, n))
lin_sd = np.array(lin_sd); mi_b = np.array(mi_b)
print(f"  一阶预测的 bootstrap 分布: mean={lin_sd.mean():+.4e} sd={lin_sd.std():.4e}")
print(f"  真实 MI 的 bootstrap 分布:  mean={mi_b.mean():+.4e} sd={mi_b.std():.4e}")
print(f"  ⇒ 若 mean(mi_b) 远负于 mean(lin_sd)，说明 MI 对 K 的**凹性**让每个 bootstrap 复制都往下掉，"
      f"\n     于是百分位区间整体位于 I_hat 之下，而真值 I_true 在 I_hat 之上 => 覆盖 0%（结构性，不是 bug）。")
