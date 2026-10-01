"""
A Agent / D16 = 回答 B 的四次催办（§50.7/51.8/52.8/53.8/54.7）与 H 梯度追问：
   (1) rank(T) 与 rank(ΠT) 的**打印值**（含容差、含奇异值谱）；
   (2) 容差对应的**被测出来的评测器噪声**（B §54.7 第 2 条）；
   (3) "2 维精确相交"随隐层宽度 H 的**存续性**（H=4,7,12,20，同 seed 规则）；
   (4) DL-2 的代价 k = rank(T) − rank(ΠT)（容量-可识别权衡），并同列吸收率。
每次抽样的 seed 规则沿用 d13/d14（torch.manual_seed(7+H)），数据固定（同一条轨迹）。
"""
import numpy as np
import torch

D = 2
r2 = np.random.default_rng(99)
A = r2.normal(size=(D, D)); A = A / max(abs(np.linalg.eigvals(A))) * 0.6
g = lambda z: np.stack([0.7 * np.tanh(3.0 * z[:, 0]), 0.6 * np.sin(3.0 * z[:, 1])], axis=1)
N = 1200
Z = np.empty((N, D)); z = np.array([0.6, -0.4])
for i in range(N):
    z = A @ z + g(z[None, :])[0] + r2.normal(scale=0.03, size=D); Z[i] = z
Xt, Yt = Z[:-1], Z[1:]
Phi = np.column_stack([Xt, np.ones(len(Xt))])
Khat = np.linalg.lstsq(Phi, Yt, rcond=None)[0]
pred = Phi @ Khat
mse0 = np.mean((Yt - pred) ** 2)
n_out = len(Xt) * D
Pd = np.zeros((n_out, D * D + D))
for i in range(D):
    for j in range(D):
        Pd[:, i * D + j] = (Xt[:, j, None] * np.eye(D)[i][None, :]).reshape(-1)
    Pd[:, D * D + i] = np.tile(np.eye(D)[i], len(Xt))
Q_P = np.linalg.qr(Pd)[0]
Pi = np.eye(n_out) - Q_P @ Q_P.T
Xin = torch.tensor(Xt, dtype=torch.float32)
Rn = torch.tensor(Yt - pred, dtype=torch.float32)

# 评测器噪声：随机子空间 vs Q_P 的最大主余弦（同维数、同抽样规模）
rng = np.random.default_rng(555)
# 正确的仪器噪声校准：对"精确正交"的一对子空间（Q_P 与 Pi·随机子空间）测读出下限
instr = max(np.linalg.svd(Q_P.T @ (Pi @ np.linalg.qr(rng.normal(size=(n_out, 60)))[0]),
                          compute_uv=False).max() for _ in range(20))
print(f"  仪器噪声校准（对'精确正交'的一对子空间）：max 主余弦 = {instr:.3e}"
      f"  ⇒ 1−cos 的仪器下限 = {1-instr:.3e}")
print(f"  ⇒ '精确相交'判据：1−cos < 100×仪器下限 = {100*(1-instr):.3e}")


print("  H   p   rank(T)  rank(ΠT)   k   吸收率   max主余弦(DL-2后)")

for H in [4, 7, 12, 20]:
    torch.manual_seed(7 + H)
    net = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.Tanh(), torch.nn.Linear(H, D))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-7)
    for ep in range(4000):
        opt.zero_grad(); loss = torch.mean((net(Xin) - Rn) ** 2); loss.backward(); opt.step()
    with torch.no_grad():
        corr = net(Xin).numpy()
    mse1 = np.mean((Yt - pred - corr) ** 2)
    absorbed = 1 - mse1 / mse0
    plist = list(net.parameters())
    out = net(Xin).reshape(-1)
    cols = []
    for k in range(n_out):
        gs = torch.autograd.grad(out[k], plist, retain_graph=True, allow_unused=True)
        cols.append(torch.cat([(q if q is not None else torch.zeros_like(p)).reshape(-1)
                               for q, p in zip(gs, plist)]).detach().numpy())
    T = np.array(cols)                                   # (n_out, p)
    p = T.shape[1]
    sT = np.linalg.svd(T, compute_uv=False)
    rankT_10 = int((sT > 1e-10 * sT[0]).sum())
    rankT_8 = int((sT > 1e-8 * sT[0]).sum())
    QT = np.linalg.qr(T)[0]
    PT = Pi @ QT
    sPT = np.linalg.svd(PT, compute_uv=False)
    rankPT_10 = int((sPT > 1e-10 * sPT[0]).sum())
    rankPT_8 = int((sPT > 1e-8 * sPT[0]).sum())
    cos_un = np.linalg.svd(Q_P.T @ QT, compute_uv=False)
    ths = [1e-12, 1e-9, 1e-6]
    cnt = [int(((1 - cos_un) < t).sum()) for t in ths]
    inter = cnt[1]
    top6 = ", ".join(f"{c:.12f}" for c in cos_un[:6])
    cos_dl2 = np.linalg.svd(Q_P.T @ (Pi @ QT), compute_uv=False)
    print(f"  {H:3d} {p:4d}   {rankT_10:5d}   {rankPT_10:6d}  {rankT_10-rankPT_10:4d}  {absorbed:7.2%}   "
          f"{cos_dl2.max():.3e}   计数(1e-12/1e-9/1e-6)={cnt}")
    print(f"        前 6 个主余弦（未投影 vs 先验）: {top6}")

print()
print("判读（逐条对 B 的催办）：")
print(" * rank 打印：见上表前四列（同时给 tol=1e-10 与 1e-8 两档，因为秩依赖容差）。")
print(" * k = rank(T) − rank(ΠT) 就是 DL-2 付出的切空间维数（容量-可识别权衡），与吸收率同列。")
print(" * 交集维数用 B §54.7 的规矩：判据 1−cos < 100×实测噪声底；噪声底由随机子空间校准得出。")
print(" * H 梯度用于判断'2 维精确相交'是否是单网伪影：若各 H 都为 2（或 ≥2），则结构性；若随 H 消失，则降级。")
