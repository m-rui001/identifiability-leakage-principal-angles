"""A/D14：诊断 [I2] 的 α_min=0 —— 是"真交集"还是"采样秩亏"？打印奇异值谱与交集维数。"""
import numpy as np, torch
torch.manual_seed(23); rng=np.random.default_rng(23)
D=2; r2=np.random.default_rng(99)
A=r2.normal(size=(D,D)); A=A/max(abs(np.linalg.eigvals(A)))*0.6
g=lambda z: np.stack([0.7*np.tanh(3.0*z[:,0]),0.6*np.sin(3.0*z[:,1])],axis=1)
N=1500; Z=np.empty((N,D)); z=np.array([0.6,-0.4])
for i in range(N):
    z=A@z+g(z[None,:])[0]+r2.normal(scale=0.03,size=D); Z[i]=z
X,Y=Z[:-1],Z[1:]
Phi=np.column_stack([X,np.ones(len(X))]); Khat=np.linalg.lstsq(Phi,Y,rcond=None)[0]; pred=Phi@Khat
for H in [16,64]:
    torch.manual_seed(7+H)
    net=torch.nn.Sequential(torch.nn.Linear(D,H),torch.nn.Tanh(),torch.nn.Linear(H,D))
    opt=torch.optim.Adam(net.parameters(),lr=3e-3)
    Xn=torch.tensor(X,dtype=torch.float32); Rn=torch.tensor(Y-pred,dtype=torch.float32)
    for ep in range(4000):
        opt.zero_grad(); l=torch.mean((net(Xn)-Rn)**2); l.backward(); opt.step()
    with torch.no_grad(): corr=net(Xn).numpy()
    print(f"H={H}: 吸收率 {1-np.mean((Y-pred-corr)**2)/np.mean((Y-pred)**2):.2%}")
    plist=list(net.parameters()); out=net(Xn).reshape(-1); n_out=out.numel()
    sel=np.arange(0,n_out,max(1,n_out//300))
    cols=[]
    for k in sel:
        gs=torch.autograd.grad(out[k],plist,retain_graph=True,allow_unused=True)
        cols.append(torch.cat([(q if q is not None else torch.zeros_like(p)).reshape(-1) for q,p in zip(gs,plist)]).detach().numpy())
    T=np.array(cols); print("  切空间矩阵",T.shape,"奇异值>1e-10 的个数",int((np.linalg.svd(T,compute_uv=False)>1e-10).sum()))
    QT=np.linalg.qr(T)[0]
    Pd=np.zeros((len(X)*D,D*D+D))
    for i in range(D):
        for j in range(D): Pd[:,i*D+j]=(X[:,j,None]*np.eye(D)[i][None,:]).reshape(-1)
        Pd[:,D*D+i]=np.tile(np.eye(D)[i],len(X))
    QP=np.linalg.qr(Pd[sel])[0]
    sv=np.linalg.svd(QP.T@QT,compute_uv=False)
    print("  前 8 个主余弦:",np.round(sv[:8],12),"  交集维数(sv>1-1e-6):",int((sv>1-1e-6).sum()))
    # 直接查：先验空间方向是否在切空间里（对单一一层权重列）
    W1=net[0].weight.detach().numpy(); b1=net[0].bias.detach().numpy(); W2=net[2].weight.detach().numpy()
    act=np.tanh(X@W1.T+b1); gate=1-act**2
    print("  门控 (1-tanh^2) 的标准差/均值:",float(gate.std()/gate.mean()))
    # 检查方向 d_{k,l}(t) = W2[:,k]*gate_k(t)*z_l(t) 是否能被"线性于 z"解释
    k=0; d=W2[:,k][None,:]*gate[:,[k]]*X          # N×D 的"梯度方向场"
    fit=np.column_stack([X,np.ones(len(X))])@np.linalg.lstsq(np.column_stack([X,np.ones(len(X))]),d,rcond=None)[0]
    print(f"  单个一阶权重方向 k=0 的'线性可解释比例' = {1-np.linalg.norm(d-fit)/np.linalg.norm(d):.4%}")

print()
print("=" * 90)
print("[补救测试] 把 MLP 的输入换成对 [z,1] 正交化的非线性特征 φ(z)，α_min 是否 > 0？")
mu = X.mean(axis=0)
Ph = np.column_stack([X[:, 0] ** 2 - (X[:, 0] ** 2).mean(), X[:, 1] ** 2 - (X[:, 1] ** 2).mean(),
                      X[:, 0] * X[:, 1] - (X[:, 0] * X[:, 1]).mean(),
                      np.sin(X[:, 0]) - np.sin(X[:, 0]).mean(), np.cos(X[:, 1]) - np.cos(X[:, 1]).mean()])
# 把 φ 对 [z,1] 做 Gram-Schmidt 正交化（数据度量）
Phi_p = np.column_stack([X, np.ones(len(X))])
Ph_orth = Ph - Phi_p @ np.linalg.lstsq(Phi_p, Ph, rcond=None)[0]
H2 = 16
torch.manual_seed(101)
net2 = torch.nn.Sequential(torch.nn.Linear(Ph.shape[1], H2), torch.nn.Tanh(), torch.nn.Linear(H2, D))
opt = torch.optim.Adam(net2.parameters(), lr=3e-3)
Pn = torch.tensor(Ph_orth, dtype=torch.float32); Rn = torch.tensor(Y - pred, dtype=torch.float32)
for ep in range(4000):
    opt.zero_grad(); l = torch.mean((net2(Pn) - Rn) ** 2); l.backward(); opt.step()
with torch.no_grad():
    corr2 = net2(Pn).numpy()
print(f"   正交特征修正器的吸收率: {1-np.mean((Y-pred-corr2)**2)/np.mean((Y-pred)**2):.2%}")
plist = list(net2.parameters()); out = net2(Pn).reshape(-1); n_out = out.numel()
sel = np.arange(0, n_out, max(1, n_out // 300))
cols = []
for k in sel:
    gs = torch.autograd.grad(out[k], plist, retain_graph=True, allow_unused=True)
    cols.append(torch.cat([(q if q is not None else torch.zeros_like(p)).reshape(-1) for q, p in zip(gs, plist)]).detach().numpy())
T2 = np.array(cols); QT2 = np.linalg.qr(T2)[0]
QP2 = np.linalg.qr(Pd[sel])[0]
sv2 = np.linalg.svd(QP2.T @ QT2, compute_uv=False)
print("   主余弦前 8:", np.round(sv2[:8], 12))
print(f"   α_min = {np.degrees(np.arccos(min(1.0, float(sv2.max())))):.4f}°   交集维数 = {int((sv2 > 1-1e-6).sum())}")

print()
print("=" * 90)
print("[补救测试 2 / 设计律 DL-2] 把修正器输出**投影到先验灵敏度空间的正交补**：交集是否恰为 {0}？")
# 在完整数据上做投影（训练时）
Qp_full = np.linalg.qr(Pd)[0]          # (N*D, D*(D+1))：先验在输出空间里的灵敏度方向
Proj = np.eye(len(X) * D) - Qp_full @ Qp_full.T
Proj_t = torch.tensor(Proj, dtype=torch.float32)
H3 = 16
torch.manual_seed(202)
net3 = torch.nn.Sequential(torch.nn.Linear(Ph.shape[1], H3), torch.nn.Tanh(), torch.nn.Linear(H3, D))
opt = torch.optim.Adam(net3.parameters(), lr=3e-3)
for ep in range(4000):
    opt.zero_grad()
    out3 = net3(Pn).reshape(-1)
    out3 = (Proj_t @ out3).reshape(-1, D)
    l = torch.mean((out3 - Rn) ** 2)
    l.backward(); opt.step()
with torch.no_grad():
    out3 = net3(Pn).reshape(-1)
    corr3 = (Proj_t @ out3).reshape(-1, D).numpy()
print(f"   投影后修正器的吸收率: {1-np.mean((Y-pred-corr3)**2)/np.mean((Y-pred)**2):.2%}")
plist = list(net3.parameters()); out = net3(Pn).reshape(-1); n_out = out.numel()
sel2 = np.arange(0, n_out, max(1, n_out // 300))
# 切空间也要先过投影再算
cols = []
for k in range(n_out):
    gs = torch.autograd.grad(out[k], plist, retain_graph=True, allow_unused=True)
    cols.append(torch.cat([(q if q is not None else torch.zeros_like(p)).reshape(-1) for q, p in zip(gs, plist)]).detach().numpy())
Tfull = np.array(cols)                     # (n_out, p)
T3 = Proj[sel2] @ Tfull                    # 投影后再取抽样行
QT3 = np.linalg.qr(T3)[0]
QP3 = np.linalg.qr(Pd[sel2])[0]
sv3 = np.linalg.svd(QP3.T @ QT3, compute_uv=False)
print("   主余弦前 8:", np.round(sv3[:8], 12))
print(f"   α_min = {np.degrees(np.arccos(min(1.0, float(sv3.max())))):.4f}°   交集维数 = {int((sv3 > 1-1e-6).sum())}")

print()
print("[核对] 在**完整输出空间**里核对 DL-2 的正交性（而不是抽样行空间）")
lhs = Qp_full.T @ (Proj @ Tfull)
print(f"   ||Qp^T (Proj T)||_F = {np.linalg.norm(lhs):.3e}  （应≈0）")
sv_full = np.linalg.svd(Qp_full.T @ (Proj @ Tfull), compute_uv=False)
print(f"   完整空间里的主余弦 max = {sv_full.max():.3e}  ⇒ α_min = {np.degrees(np.arccos(min(1.0,float(sv_full.max())))):.2f}°")
print("   结论：DL-2 在完整空间（=同一数据度量）里给出**精确正交**；")
print("   我上一步报的'交集维数 1'是'抽样行空间'与'完整空间'两种度量的错配造成的，正确诊断是全空间版。")

print()
print("[对照] 同一批切空间矩阵，在**完整空间**里与先验空间的主角度（先正交化再算）")
QT_full = np.linalg.qr(Tfull)[0]
sv_u = np.linalg.svd(Qp_full.T @ QT_full, compute_uv=False)
print("   未投影：主余弦前 6:", np.round(sv_u[:6], 12))
print(f"   未投影 α_min = {np.degrees(np.arccos(min(1.0,float(sv_u.max())))):.4f}°  "
      f"交集维数(sv>1-1e-9) = {int((sv_u>1-1e-9).sum())}   （先验空间维数 = {Qp_full.shape[1]}）")
