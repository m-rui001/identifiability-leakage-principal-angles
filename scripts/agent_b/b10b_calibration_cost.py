"""
B Agent / B-D10b: 校准的代价 —— 把分位数从名义 0.95 抬到 q*(=0.972) 要花多少宽度？
只做 N=100,T=200 一档（W1 里欠覆盖最严重的那档）。
"""
import numpy as np

rng = np.random.default_rng(777002)


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
N, T, TRIALS, B = 100, 200, 60, 400

lev = [0.90, 0.95, 0.972, 0.99]
abserr, Dm, raw = [], {l: [] for l in lev}, []
for _ in range(TRIALS):
    Zs, Zn = [], []
    for _ in range(N):
        z = rng.normal(size=d); traj = [z]
        for _ in range(T - 1):
            z = Ktrue @ z + rng.normal(scale=np.sqrt(sigma2), size=d); traj.append(z)
        X = np.array(traj); Zs.append(X[:-1]); Zn.append(X[1:])
    Z = np.vstack(Zs); Znext = np.vstack(Zn)
    Khat = np.linalg.lstsq(Z, Znext, rcond=None)[0].T
    Mh = grad(Khat, C, Sig, n)
    I_hat = mi_lin(Khat, C, Sig, n)
    st = np.empty(B)
    for b in range(B):
        ib = rng.integers(0, len(Z), len(Z))
        Kb = np.linalg.lstsq(Z[ib], Znext[ib], rcond=None)[0].T
        st[b] = abs(float(np.sum(Mh * (Kb - Khat))))
    abserr.append(abs(I_hat - I_true)); raw.append(st)
    for l in lev:
        Dm[l].append(np.quantile(st, l))
abserr = np.array(abserr)
print("=" * 100)
print(f"[X1] N={N},T={T}, trials={TRIALS}, B={B}   median|δI| = {np.median(abserr):.3e}")
print("  分位数   覆盖率   宽度中位(|δI|的倍数)   相对 0.95 的宽度代价")
w95 = np.median(np.array(Dm[0.95]) / abserr)
for l in lev:
    cov = np.mean(abserr <= np.array(Dm[l])) * 100
    w = np.median(np.array(Dm[l]) / abserr)
    print(f"  q={l:.3f}   {cov:5.1f}%      {w:6.2f}x                {w/w95:6.3f}")
print()
print("[X2] 逐 trial 的 |δI| 在自身 bootstrap 分布中的分位位置（校准图的对角线检验）")
pos = np.array([np.mean(st < e) for st, e in zip(raw, abserr)])
for p in [0.5, 0.8, 0.9, 0.95, 0.99]:
    print(f"  名义 {p:.2f} 分位  ->  经验位置 = {np.quantile(pos, p):.3f}")
print(f"  超过 0.95 的 trial 比例 = {np.mean(pos > 0.95)*100:.1f}%   (bootstrap 在 **尾部 trial** 上失准)")
big = np.argsort(-abserr)[:6]
print("  最超出的 6 个 trial: |δI| =", np.array2string(abserr[big], precision=3),
      " 其分位位置 =", np.array2string(pos[big], precision=3))
print("  => 若超出的 trial 同时是 |δI| 最大的，说明是 **异方差/leverage** 失准，")
print("     正确修法不是抬全局分位数，而是相对化证书（除以 Δ 自身，做 conformal 比值）。")
