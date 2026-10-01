# Koopman 算子的谱、函数空间中的谱展开与状态空间几何（Mezić）

原题：Spectrum of the Koopman Operator, Spectral Expansions in Functional Spaces, and State-Space Geometry。Igor Mezić（UCSB）。arXiv:1702.07597，DOI: 10.1007/s00332-019-09598-5（J. Nonlinear Sci. 2019）。通道：TeX 源，定位用章节/公式号；抽文含作者修订标记与注释，引用编号丢失，见疑点。

## 论证纲领：谱对象 ↔ 状态空间几何 ↔ 数据谱类型的三向对应

引言把论文的问题意识立在三处：(1) 共轭性问题——von Neumann 与 Kolmogorov 已否定了"Koopman 谱等价 $\Rightarrow$ 共轭"（其反例具有混合或连续谱），本文在耗散系统上探索谱与共轭的关系，并研究从平衡点到准周期性的各类渐近行为的谱型；(2) 几何问题——特征函数的水平集承载状态空间几何（不变集、isochron、isostable 均可定义为特征函数的水平集），本文把这一关系扩展到 center/center-stable/center-unstable 流形的**联合零水平集**定义，把这类不变流形的观点从局部的"与线性子空间相切"转变为全局的、基于水平集的定义；(3) 函数空间问题——耗散系统的复合算子通常非正规、可有广义特征函数；作者选择"$L^2$ 于吸引子上、解析于吸引子外"的观测函数类，发现准周期系统的谱支撑在复平面离散集上，并由此构造两类新空间：以 Koopman 主特征函数定义的**调制 Fock 空间**与 **AKHS（Averaging Kernel Hilbert Space，RKHS 的修正）**。数据侧的落点是：所有被研究系统的谱都是**格型（lattice type）**——$n$ 个主特征值的整系数线性组合（$n$ 为状态空间维数），由此定义数据的"主相干维数"。

## 线性系统：Kato 分解、广义特征函数与子空间的水平集刻画

**简单谱情形**。$\dot x=Ax$ 的 Koopman 特征函数是 $\phi_j(x)=\langle x,w_j\rangle$（$w_j$ 为 $A^*$ 的特征向量，$A^*w_j=\lambda_j^cw_j$），由 $\dot\phi_j=\langle Ax,w_j\rangle=\langle x,A^*w_j\rangle=\lambda_j\phi_j$ 直接得出。谱展开 $U^tx=\sum_j e^{\lambda_jt}\phi_j(x_0)v_j$ 中，作者强调一个概念区分：$v_j$ **不属于 Koopman 算子而属于观测**——换成观测 $y=Cx$ 时特征值与特征函数不变、Koopman 模变为 $Cv_j$；这一性质在非线性情形保留（展开变为无穷且可能有连续谱部分）。

**一般情形（重特征值）**。用 Kato 分解 $U=\sum_h\lambda_hP_h+D_h$（$P_h$ 为代数特征空间上的投影、$D_h$ 幂零，$D_hD_k=\delta_{hk}D_h$、$P_hD_k=0$），作者给出 $e^{At}$ 的展开
$$e^{At}=\sum_{h=1}^s\Big(e^{\lambda_ht}P_h+\sum_{j<m_h}\frac{t^je^{\lambda_ht}}{j!}D_h^j\Big),$$
并以此给出 Hirsch–Smale 定理的"简单优雅"证明。对几何特征空间维数为 1 的情形，取广义特征向量链 $A\v_h^i=\lambda_h\v_h^i+\v_h^{i-1}$ 与对偶链 $A^*w_h^i=\lambda_h^cw_h^i+w_h^{i+1}$（$i<m_h$），则 $\phi_h^i=\langle x,w_h^i\rangle$ 满足 $\dot\phi_h^i=\lambda_h\phi_h^i+\phi_h^{i+1}$——即 $(d/dt-\lambda_hI)^{m_h}\phi=0$ 的零空间元素，称为 Koopman 算子在 $\lambda_h$ 的**广义特征函数**，其时间演化为 $\phi_h^i(t)=\sum_{n=0}^{m_h-i}\frac{t^n}{n!}e^{\lambda_ht}\sum_{l=m_h}^{i+n}\phi_h^l(0)$。一个 remark 记录了与普通特征函数的关键差别：**广义特征函数的乘积不再是广义特征函数**。比较 Kato 展开与特征函数展开得 $P_hx=\sum_k\phi_h^k(x)\v_h^k$、$D_h^ix=\sum_{k>i}\phi_h^k(x)\v_h^{k-i}$。

**标准形**。广义 Koopman 特征函数向量 $\boldsymbol\phi$ 满足 $\dot{\boldsymbol\phi}=J\boldsymbol\phi$（$J$ 为 Jordan 标准形，复特征值用极坐标 $(y_j,y_{j+1})=(r_j\cos\theta_j,r_j\sin\theta_j)$ 化为实块 $C_i=\begin{bmatrix}\sigma_i&\omega_i\\-\omega_i&\sigma_i\end{bmatrix}$）。推论：任何满足 $\dot{\boldsymbol\phi}=J\boldsymbol\phi$ 的函数组都是 $\dot x=Ax$ 的（广义）特征函数——这是后文用共轭定理识别非线性系统 Koopman 特征函数的工具。

**子空间 = 联合零水平集（命题 1）**。将特征值按实部排序（$u$ 个正、$c$ 个零、$s$ 个负），则
$$L_s=\{x\mid\phi_1(x)=0,\dots,\phi_{u+c}(x)=0\}=E^s,\quad L_c=\{x\mid\phi_1=\cdots=\phi_u=0,\ \phi_{u+c+1}=\cdots=\phi_{u+c+s}=0\}=E^c,\quad L_u=\{x\mid\phi_{u+c+1}(x)=0,\dots\}=E^u.$$
证明思路：置零正/零实部特征函数即消去展开中不衰减的项。作者特别指出该刻画**可推广到非线性系统**（水平集定义），而经典的"特征向量张成"定义不能（对 $\dot x=Ax+\epsilon f$ 只能得到相切性）。

## 共轭性与"开特征函数"

**因子共轭**。定义比经典拓扑共轭更广：$\mathfrak h(S^tx)=T^t(\mathfrak h(x))$，允许 $m\ne n$（$m<n$ 时含半共轭）。核心引理：若 $\phi$ 是 $U_T^t$ 在 $\lambda$ 的特征函数，则 $\phi\circ\mathfrak h$ 是 $U_S^t$ 在 $\lambda$ 的特征函数（三行等式链直接验证）；广义特征函数同样被共轭保持。推论：若能找到非线性系统到线性系统的全局共轭，则 Koopman 谱可由不动点处线性化的谱确定。

**开特征函数（open eigenfunctions）**。经典线性化定理只在平衡点/不变环面等特殊集合的邻域内有定义，本文提出把"局部"特征函数沿流延拓：$\phi:A\to\C$（$A$ 非不变集）满足 $U^\tau\phi=e^{\lambda\tau}\phi$ 对 $\tau\in I_x=(\tau^+(x),\tau^-(x))$ 称为开特征函数；$A$ 为真不变子集时称子域特征函数。关键引理：对 $A$ 上满足 $\dot\phi=\lambda\phi$ 的连续函数，用击中时间 $t(\z)$ 定义
$$\phi(\z)=e^{-\lambda t(\z)}\phi(S^{t(\z)}\z),\qquad \z\in P=B\cup A\cup F,$$
则 $\phi$ 是 $P$ 上的连续开特征函数（证明由 $t(S^\tau\z)=t(\z)-\tau$ 直接展开；连续性证明走流连续 + 指数因子控制）。一个 remark 揭示了谱选择机制：在任意非平衡点附近可把流拉直（$\dot x_1=1,\dot x_j=0$），构造 $\chi=ce^{\lambda x_1}\chi_1(x_2,\dots,x_n)$ 对**任意** $\lambda$ 都是局部开特征函数——因此"状态空间中的奇点（不动点）与回复性（极限环附近的环绕）才是 Koopman 谱中 $\lambda$ 的选择者"（原文明说的结构性观察；这一概念后被 Korda–Mezić 2020 用于预测与控制，见本批次另一篇推理链）。

## 全局稳定平衡的非线性系统：从 Poincaré 线性化到全局中心流形

**Poincaré 线性化与特征模展开**。对 $\dot x=\F(x)=Ax+v(x)$、非共振（或 Siegel 条件 $|\lambda_s-\sum_km_k\lambda_k|\ge C/(\sum_km_k)^\nu$）的解析场，Poincaré 定理给出解析共轭 $\mathfrak h$。以 $s_j(x)=\phi_j(\mathfrak h(x))$ 为新坐标，解析观测 $f$ 经 Taylor 展开得多重指标展开
$$\mathbf f(x)=\sum_{\{k_1,\dots,k_n\}\in\mathbb N^n}\overline{\mathbf v}_{k_1\cdots k_n}\,s_1^{k_1}(x)\cdots s_n^{k_n}(x),\qquad U^t\mathbf f(x)=\sum_{\{k\}}\cdots e^{(k_1\lambda_1+\cdots+k_n\lambda_n)t}.$$
$(0,\dots,0)$ 模是时间平均；$Ds_i(0)=\langle\cdot,w_i\rangle$ 即 $D\F|_0$ 的左特征向量，故 $s_i(x)=\langle x,w_i\rangle+o(\|x\|)$——近平衡处特征函数由线性化系统近似。命题（用延拓引理 + 广义 Laplace 分析）：若 Taylor 展开在全吸引盆 $\mathcal B(0)$ 有效，则展开在 $\mathcal B(0)$ 上成立，$\tilde s_j(\z)=e^{-\lambda_jt(\z)}s_j(S^{t(\z)}\z)$，模按 $\tilde v_{k}=e^{-(k\cdot\lambda)t(\z)}\overline v_k$ 变换。作者由此点出全文的一个"striking realization"：线性与非线性平衡系统在算子表示上的**唯一**差别是展开的有限 vs 无穷；非线性展开还是渐近性的（不同项携带不同衰减/增长率）。

**Hartman–Grobman 与全局线性化**。Hartman 定理只需 $C^2$、特征值实部非零，无共振条件：存在 $C^1$ 微分同胚 $\tilde{\mathfrak h}$ 使 $U^t\tilde{\mathfrak h}(x)=e^{At}\tilde{\mathfrak h}(x)$（局部、有限时间）。对角化后 $\tilde\kappa=V^{-1}\tilde{\mathfrak h}$ 的分量即 Koopman 特征函数。**自治流线性化定理**（引作者先前结果）把它全局化：$A$ Hurwitz 时，存在 $C^1$ 微分同胚 $\mathfrak h:\mathcal B\to\mathbb R^n$、$D\mathfrak h(0)=I$、$\dot y=Ay$ 在全吸引盆成立——证明用 Lyapunov 水平面上的截面 $\Sigma$（由反 Lyapunov 定理保证每条轨迹唯一穿越）做 $\mathfrak h(x)=e^{-At(x)}\tilde{\mathfrak h}(S^{t(x)}(x))$。鞍点情形用延拓引理得开特征函数；推论给出 $W^s_P$、$W^u_P$（与局部流形相连的"主干"部分）的联合零水平集刻画。

**中心流形（§5.4）**。用 Palmer 线性化定理（$C^0$ 共轭、块分解 $B$ 零实部块/$A$ 非零实部块、$g,h$ 有界 Lipschitz），命题：不稳定广义特征函数的联合零水平集 $L_{cs}$ 是 center-stable 流形、稳定 + 不稳定全部置零的 $L_c$ 是**全局唯一** center 流形、$L_{cu}$ 是 center-unstable。remark 处理两个关键反对意见：(1) 全局性——有界性要求并不阻碍，因为若稳定/不稳定流形带相配的纤维化与交换流的投影 $\Pi^s$，则 $\mathfrak g=\mathfrak h\circ\Pi^s$ 给出半共轭，从而构造出稳定/不稳定特征函数；(2) **Kelley 经典反例** $\dot x_1=x_1^2,\dot x_2=x_2$（局部中心流形不唯一：所有 $x_2=Ae^{-1/x_1}$，$x_1>0$ 与 $x_2=0$（$x_1\le0$）的并都相切于中心子空间）——但这些曲线在 $t\to-\infty$ 有指数行为、不符合全局中心流形性质；算子观点下全局中心流形是 $x_2=0$（特征函数 $f_2=x_2$ 的零水平集）。作者还展示该例中 $f_1=e^{1/x_1}$（$x_1<0$）是特征值 $-1$ 的无穷光滑但**不解析**的特征函数，其零水平集 $x_1\ge0$ 是（带边界的）不稳定流形；$f_3=e^{-1/x_1}$（$x_1\ge0$）对应特征值 1；乘积 $f=x_2e^{1/x_1}$ 的水平集是 $x_1\ge0$ 的轨迹。

## 耗散系统的函数空间：张量积构造与谱定理

设 $\mathcal A$ 为零 Lebesgue 测度的全局 (Milnor) 吸引子、带物理测度 $\mu$。$\mathcal H_\mathcal A=L^2(\mathcal A,\mu)$ 上 $U^t$ 酉；用 Sell 型坐标化（$\dot y=A(\mathbf u)y+\v$、$\dot{\mathbf u}=\boldsymbol\omega(\mathbf u)+\boldsymbol\Omega$，法向双曲 + 谱条件下 $C^s$ 共轭于斜线性系统 $\dot y=A(\mathbf u)y$、$\dot{\mathbf u}=\boldsymbol\omega(\mathbf u)$）+ 截面 $\Sigma$ 拉回，全吸引盆的每点都有 $(\mathbf u,y)$ 坐标。取正交补 $\tilde{\mathcal H}_\mathcal B=\{f\in C(\mathcal D)\mid\int_\mathcal D f\phi\,d\mu=0,\ \forall\phi\in\mathcal H_\mathcal A\}$（remark 引 Giannakis 的论证：$\mu$ 为有限 Borel 测度时可换成"$f$ 在 $\mathcal A$ 上为零"）。引理：$\tilde{\mathcal H}_\mathcal B$ 在 $U^t$ 下不变（用 $\mu$ 不变性）。定义 $\mathcal H=\mathcal H_\mathcal A\otimes\mathcal H_\mathcal B$（$\mathcal H_\mathcal B=\tilde{\mathcal H}_\mathcal B\cup\mathbf 1$），假设 $U^t|_{\mathcal H_\mathcal B}$ 为标量型谱算子，则**定理（张量积谱）**：
$$\sigma(U^t)=\mathrm{cl}\big(P(\sigma(U^t|_{\mathcal H_\mathcal A}),\sigma(U^t|_{\mathcal H_\mathcal B}))\big),\qquad U^t=\int_\mathbb C\int_\mathbb R e^{zt}e^{i2\pi\omega t}\,dP_\omega\,dP_z.$$
一维示例：$\dot x=F(x)$（全局稳定不动点）时 $\mathcal H_\mathcal A$ 是常函数空间、$\tilde{\mathcal H}_\mathcal B$ 恰为 **Fock 空间**（$\int_\C|f|^2e^{-|z|^2}dz<\infty$、内积 $\sum a_nb_n^cn!$、再生核 $K(z,w)=e^{z\cdot w}$、标准正交基 $e_n=z^n/\sqrt{n!}$）。

## 极限环与准周期吸引子的谱展开

**$\mathbb R^2$ 极限环（定理，作者注明完整证明首次给出）**。吸引盆内可化为 $\dot y=A(s)y$、$\dot s=1$（$s\in S^1$，$A(s)$ 为 $2\pi$ 周期标量函数）。引理：$g(y,s)=b(s)y^\alpha$、$b(s)=ce^{-\alpha a^f(s)}$、$a^f(s)=\int_0^s(A(\bar s)-A^*)d\bar s$ 是特征函数，特征值 $e^{\alpha A^*t}$（$A^*=\frac{1}{2\pi}\int_0^{2\pi}A(z)dz$；证明解 PDE $\frac{\partial g}{\partial y}A(s)y+\frac{\partial g}{\partial s}=\dot\mu(0)g$，再用 $b$ 的周期性在 $[0,2\pi]$ 上积分得 $\dot\mu(0)=\alpha A^*$）。定理：任何 $y$ 解析、$s$ 中 $L^2$ 的 $F(y,s)$ 有展开
$$F(y,s)=\sum_{m=0}^\infty\sum_{n=-\infty}^\infty a_{mn}\,y^me^{-m\int_0^s(A(\bar s)-A^*)d\bar s}e^{ins},$$
特征值格 $\{mA^*+in\}$。证明走 $y$-Taylor + 系数函数按 $a_m(s)=e^{-m\int_0^s(A-A^*)}\bar a_m(s)$ 分解 + Fourier。推论（原 $\color{black}$ 新增）：任何全局渐近稳定到极限环的 $C^2$ 二维系统可写为 $\dot r=A^*r$、$\dot\theta=\omega$（令 $r=e^{-\int_0^s(A-A^*)}y$）。AKHS 在此显式构造：$\tilde{\mathcal H}_\mathcal B=\{\sum_{m\ge1}f_m(s)y^m,\ \frac{1}{2\pi}\sum m!\int f_m^2<\infty\}$，核 $K(y,z)=\sum y^mz^m/m!$，$\langle K_z(y),F\rangle=\frac{1}{2\pi}\int_{S^1}F(s,y)ds=\bar F$（"平均"核之名由此）；$A(s)\le0$ 时谱为 $\mathrm{cl}(\{mA^*+in\})$。

一个重要 remark 区分"谱展开"与"谱本身"：以 $S^1$ 上无理旋转为例，特征值 $\{e^{in2\pi\omega}\}$ 稠密于 $S^1$（谱为 $S^1$），但 $L^2$ 展开只用特征值的稠密子集即可——谱展开不必用全部谱。

**$\mathbb R^n$ 极限环**。$\dot y=A(s)y$、$\dot s=1$（$A(s)$ 为 $2\pi$ 周期矩阵）。Floquet 理论：$\zeta(\mathbf y,t)=V^{-1}P^{-1}(t)\mathbf y$（$P(s)$ Floquet 周期矩阵、$B$ 稳定性矩阵、$V$ 对角化阵）是主 Koopman 特征函数，特征值 $e^{(\mathbf m\cdot\boldsymbol\mu+ik)t}$（$\boldsymbol\mu$ 为 Floquet 指数）。作者指出 $n$ 维情形**无法显式写出特征函数**——本质原因是 $A(s_1)$ 与 $A(s_2)$ 矩阵不可交换（与标量情形的本质差异，原文明说）。AKHS 的 $n$ 维推广（权 $\kappa!$、核 $\sum(\zeta w)^\kappa/\kappa!$）给出谱 $\mathrm{cl}(\{\lambda_{\mathbf m,k}\})$。

**准周期吸引子**。$\mathbf m$ 维环面、频率向量满足 Diophantine 条件 ${\bf k}\cdot\boldsymbol\omega\ge c/|{\bf k}|^\gamma$；斜线性系统 $\dot y=A(\boldsymbol\theta)y$、$\dot{\boldsymbol\theta}=\boldsymbol\omega$。在"准 Floquet 谱满"（由无指数二分性定义）与 $A(\boldsymbol\theta+\boldsymbol\omega t)$ 光滑的条件下存在准 Floquet 变换 $P(t)$ 化为常矩阵 $B$。定理：$\zeta(\mathbf y,\boldsymbol\theta)=V^{-1}P^{-1}(\boldsymbol\theta)\mathbf y$ 为主特征函数，展开 $\mathbf G(\mathbf y,\boldsymbol\theta)=\sum_{\mathbf m,\boldsymbol\kappa}\mathbf a_{\mathbf m\boldsymbol\kappa}\,\zeta^{\mathbf m}e^{i\boldsymbol\kappa\cdot\boldsymbol\theta}$，特征值 $e^{(\mathbf m\cdot\boldsymbol\mu+i\boldsymbol\kappa\cdot\boldsymbol\omega)t}$。remark 指出与平衡情形的对比：斜线性系统的谱展开只需 Diophantine 非共振（多角变量的情形），但完整的非线性谱展开仍需 Poincaré 型非共振条件以保证共轭在离吸引子方向解析。**Isostables 推广**：设 $|\mu_k|>\dots>|\mu_1|>0$、实部 $\sigma_n<\dots<\sigma_1<0$，则 $z_1$ 的水平集定义为 isostable（同一条 isostable 上的初值以速率 $\sigma_1$ 同时收敛到吸引子）；$e^{is(x)}$ 的水平集是 isochron、$e^{i\theta_j(x)}$ 是准周期情形的广义 isochron。稳定/不稳定/中心流形在同设置下由主特征函数（$\mathbf m=(0,\dots,1_j,\dots,0)$ 型）的联合零水平集定义。

## 数据侧：主相干维数与连续谱警示

**主相干维数**。全部上述系统的谱都是格型 $\lambda_{\mathbf n,\boldsymbol\kappa}=\mathbf n\cdot\boldsymbol\mu+i\boldsymbol\kappa\cdot\boldsymbol\omega$（$j$ 为吸引子维数、$m=n-j$）。据此定义：若实验/数值观测到的谱中 $n$ 个主特征值生成其余点谱，则数据的主相干维数为 $n$。示例：三维极限环系统 $\dot x=y,\dot y=x-x^3-cy,\dot\theta=\omega$，$c=\sqrt7,\omega=1$ 时平衡点 $\pm1$ 处线性化特征值 $\lambda_{3,4}=-1.3228756\pm0.5i$，另两个主特征值为 $\pm i$；对 $\{0,\dots,4\}\times\{0,\dots,4\}$ 整数格画谱点。格谱可用 DMD 变体或紧化方法从数据逼近。

**连续谱的警示故事**。测度保持的非耗散例：可积单自由度系统 $\dot I=0,\dot\theta=I$（action-angle）。$I$ 是特征值 0 的特征函数，但**没有**其他特征值：$\phi_\omega$ 须满足 $\phi_\omega(I,\theta+It)=e^{i\omega t}\phi_\omega(I,\theta)$，写 $\phi_\omega=re^{i\varphi}$ 只能对 $\omega=I$ 成立——而对每个 $I$ 取值的 $\omega=I$ 说明特征函数在严格意义下不存在。定义 $\phi(I,\theta)=e^{i\theta}\delta(I-c)$ 只弱意义满足、是测度不是函数；族 $\phi_j=e^{ij\theta}\delta(I-c)$ 称为**特征测度**（eigenmeasures）。连续谱被理解为"特征函数被特征测度取代的点谱"：对 $f(I,\theta)=\sum_ja_j(I)e^{ij\theta}$，演化 $U^tf=\sum_je^{ijIt}a_j(I)e^{ij\theta}$，可写成 $U^tf=f^*(I)+\int_\mathbb Re^{i\beta t}dP_\beta(f)$，其中 $dP_\beta(f)=\sum_{j\ne0}a_j(I)e^{ij\theta}\delta(jI-\beta)d\beta$ 构成投影值测度的"微分"（作者验证 $P(\mathbb R)=I$ 及 $\mu(A)$ 是绝对连续测度）。实验含义：单摆实验（分界线内初值）的单轨迹 Fourier 谱在频率 $jI_0$ 处有峰，换初值峰位置连续变化——**单轨迹测得的"谱"与 Koopman 算子的（连续）谱是两回事**；二者只在吸引子上混合等更强动力学的情形可能重合，一般而言谱对同一遍历分量内几乎所有初值相同、对不同遍历分量不同。

## 结论中的方法论警示

结论除总结外给出一处对数据驱动方法的直接影响（原文明说）：**自然适合耗散线性系统的 Fock 空间在非线性下不封闭**——这可解释以多项式为 EDMD 基时谱"切换"的现象（轨迹从一个奇异解走向另一个时，函数空间不稳定、动力学"漏出"子空间）；常规 RKHS 方法通常回避 RKHS 在动力学作用下的封闭性问题，同样可能"漏出"；以主特征函数构造的调制 Fock 空间不存在该问题（动力学保持在不变子空间内），Hankel DMD 方法也有好的不变性。结果可适当修改（共振条件）后推广到离散时间映射。

## 推理重建（标明为推断）

- 引言宣称的"特征值格 $\to$ 主相干维数"作为数据分析工具，其可用性依赖数据方法（DMD/紧化）恢复的是点谱的近似；论文只给了格结构示例图，没有给出从含噪数据确定主特征值个数的算法，属方向性声明。
- 连续谱例中"单轨迹峰谱 vs 算子连续谱"的区分，其与实验中常见"谱随初值变化"现象的对应是作者的阐释性推断（原文用 "could coincide ... expected" 等措辞）。

## 限定条件与边界（原文显式）

- Poincaré 展开需要解析场 + 非共振/Siegel 条件；Hartman 只需 $C^2$ + 实部非零但只在局部（鞍点情形有限时间）。
- 张量积谱定理假设 $U^t|_{\mathcal H_\mathcal B}$ 为标量型谱算子（quasinilpotent 部分设为零）。
- 斜线性展开要求准 Floquet 谱"满"（$m$ 个孤立点、无指数二分性）；Diophantine 条件只对多角变量情形需要。
- $n$ 维极限环的特征函数无显式表达式（$A(s_1),A(s_2)$ 不可交换）。
- 全部展开针对"$L^2$ 于吸引子、解析于离吸引子方向"的观测类；观测空间更大或动力学更复杂时谱可支撑在非离散集上（引言明说）。

## 疑点

- 抽文所有 `\cite` 编号丢失：Gaspard 等的谱展开、Sell 吸引子坐标化、法向双曲共轭结果、Korda–Mezić 的受控 Koopman 方案、Kelley 例的出处（脚注有说明）等均无法核对。
- §8 展开式 $F(y,s)=\sum_{\k,|\k|\ge1}f_\k(s)\zeta^{\mathbf m}$ 中下标 $\k$ 与指数 $\mathbf m$ 混用，且定理 3（quasiperiodic）中 $\zeta^{\mathbf m}$ 的定义写为 "$z_1^{m_1}(\mathbf y,\boldsymbol\theta)\cdots z_1^{m_k}(\mathbf y,\boldsymbol\theta)e^{i\boldsymbol\kappa\cdot\boldsymbol\theta}$"，最后一项应为 $z_k^{m_k}$，疑原文笔误，照录并标疑。
- 频率向量 Diophantine 条件 ${\bf k}\cdot\boldsymbol\omega\ge c/|{\bf k}|^\gamma$ 照录（严格版本通常取绝对值）。
- $\mathcal H=\mathcal H_\mathcal A\otimes\mathcal H_\mathcal B$ 的正文中 $\mathcal H_\mathcal B$ 定义为 $\tilde{\mathcal H}_\mathcal B\cup\{\mathbf 1\}$（并上常值函数），而张量分解中 $\mathcal H_\mathcal A$ 已含常值函数，两者在常函数上的重叠如何处理原文未展开，读谱定理时需注意。
- 摘要与正文出现的 "Modulated Fock Space" 与结论中的 "Modified Fock Space" 拼写不一致，疑为笔误。
