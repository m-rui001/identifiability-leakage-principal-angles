"""A/D16b：判定"修正器切空间吃掉先验灵敏度空间"是否**随先验变富而加剧**（= 与 B 的谱退化机制区分开）。
   (a) 先验 [z,1]（dim 6）；(b) 先验 [z,1,z1²,z2²,z1z2]（dim 10）。同一数据、同一 MLP(H=16)。
   报：每个先验基方向对修正器切空间的主余弦（升序前 8 个）与 计数(1−cos<1e-6 / 1e-9)。"""
import numpy as np, torch
D=2; r2=np.random.default_rng(99)
A=r2.normal(size=(D,D)); A=A/max(abs(np.linalg.eigvals(A)))*0.6
g=lambda z: np.stack([0.7*np.tanh(3.0*z[:,0]),0.6*np.sin(3.0*z[:,1])],axis=1)
N=1200; Z=np.empty((N,D)); z=np.array([0.6,-0.4])
for i in range(N):
    z=A@z+g(z[None,:])[0]+r2.normal(scale=0.03,size=D); Z[i]=z
Xt,Yt=Z[:-1],Z[1:]
torch.manual_seed(7+16)
net=torch.nn.Sequential(torch.nn.Linear(D,16),torch.nn.Tanh(),torch.nn.Linear(16,D))
opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-7)
Phi0=np.column_stack([Xt,np.ones(len(Xt))]); pred0=Phi0@np.linalg.lstsq(Phi0,Yt,rcond=None)[0]
Xin=torch.tensor(Xt,dtype=torch.float32); Rn=torch.tensor(Yt-pred0,dtype=torch.float32)
for ep in range(4000):
    opt.zero_grad(); l=torch.mean((net(Xin)-Rn)**2); l.backward(); opt.step()
plist=list(net.parameters()); out=net(Xin).reshape(-1); n_out=out.numel()
cols=[]
for k in range(n_out):
    gs=torch.autograd.grad(out[k],plist,retain_graph=True,allow_unused=True)
    cols.append(torch.cat([(q if q is not None else torch.zeros_like(p)).reshape(-1) for q,p in zip(gs,plist)]).detach().numpy())
T=np.array(cols); QT=np.linalg.qr(T)[0]

def build_Pd(feats):
    F=feats.shape[1]; Pd=np.zeros((n_out,F*D))
    for i in range(D):
        for j in range(F):
            Pd[:,i*F+j]=(feats[:,j,None]*np.eye(D)[i][None,:]).reshape(-1)
    return Pd

for tag,feats in [("[z,1]（dim 6）",Phi0),
                  ("[z,1,z1²,z2²,z1z2]（dim 10）",np.column_stack([Xt,np.ones(len(Xt)),Xt[:,0]**2,Xt[:,1]**2,Xt[:,0]*Xt[:,1]]))]:
    Pd=build_Pd(feats); QP=np.linalg.qr(Pd)[0]
    cos=np.linalg.svd(QP.T@QT,compute_uv=False)
    n=len(cos)
    print(f"  {tag}: 先验空间 dim={QP.shape[1]}，主余弦（降序前 8）=",np.round(cos[:8],10))
    print(f"      计数(1−cos<1e-6) = {int(((1-cos)<1e-6).sum())}/{n}；"
          f"(1−cos<1e-9) = {int(((1-cos)<1e-9).sum())}/{n}；最小余弦 = {cos.min():.8f}")
print()
print("判读：若先验变富（dim 6→10）后'被吃掉'的维数也随之变多（或至少不减少），")
print("      则这是**架构性**的（修正器一阶层里含线性于 z 的方向），与先验算子谱的退化无关；")
print("      因此它与 B 的'顶特征值重合 ⇒ v₁ 不被数据决定'（谱退化）是**不同机制**，")
print("      两者的共同形状是'通道重合 ⇒ 分解不被数据决定，控制量是相应间隙'。")
