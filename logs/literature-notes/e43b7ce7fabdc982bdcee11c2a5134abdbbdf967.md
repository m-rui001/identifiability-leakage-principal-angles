# 基于 Koopman 算子的鲁棒管式模型预测控制（r-KMPC，Zhang, Pan, Scattolini, Yu & Xu）

原题：Robust tube-based model predictive control with Koopman operators（Extended Version）。作者单位：国防科大、代尔夫特理工、米兰理工、吉林大学。arXiv:2108.13011，DOI: 10.1016/j.automatica.2021.110114（Automatica 2022）。通道：TeX 源（扩展版），定位用章节/公式号；抽文中大量 `\color{black}`/`\xl{}` 修订标记与注释块为作者修改痕迹，引用编号丢失，见疑点。

## 论证的缺口与双重贡献

被控对象是非线性离散系统 $x^+=f(x,u)+w_o$（$w_o\in\mathcal{W}_o$ 有界加性扰动、可未知不可测；$f$ 可部分或完全未知，$C^\infty$、$f(0,0)=0$），目标是最小化二次代价 $J=\sum_{k=0}^{+\infty}\|x_k\|_Q^2+\|u_k\|_R^2$。引言的论证路径：非线性 MPC 在线问题是非凸的、计算昂贵；Koopman 有限维截断给出线性预测器，使"非线性系统的线性 MPC"成为可能（Korda & Mezić 的 KMPC、集成 Koopman、深度学习 Koopman 模型、流控应用等），但作者指出此前 Koopman MPC 的两个未解决问题（原文明说）：(i) 建模误差不可避免（有限维逼近 + 扰动估计误差），$C\hat s\in\mathcal{X}$ 并不保证真实状态约束 $x\in\mathcal{X}$ 满足；(ii) 建模误差与扰动下的闭环鲁棒性未被验证。由此给出两点贡献：一是从数据出发的线性鲁棒 Koopman MPC 设计（无需显式模型，且最终控制律是非线性的）；二是在"对提升观测函数施加标准先验条件"下证明闭环鲁棒性与名义逐点收敛——这允许使用**有限阶** Koopman 模型而不需要逼近算子的收敛性（摘要强调：不假设 $\mathcal{K}_{N_\phi}$ 收敛到 $\mathcal{K}$）。

## §2 预备：受控 Koopman 算子的扩展状态处理与数据驱动线性预测器

自治系统的 Koopman 算子 $\mathcal{K}\phi=\phi\circ f(\cdot,0)$ 被推广到受控情形时，作者采用"扩展状态"方案：$\boldsymbol\chi=(x,\boldsymbol{u}_w)$，$\boldsymbol{u}_w=\{u_w(i)\}_{i=0}^{+\infty}$ 为控制与扰动组成的无穷序列（$u_w(i)=(u(i),w_o(i))$），$\boldsymbol\chi^+=F(\boldsymbol\chi):=(f_W(x,\boldsymbol{u}_w(0)),\Gamma\boldsymbol{u}_w)$，$\Gamma$ 为左移算子；Koopman 算子定义在扩展观测空间 $\mathcal{F}_e$ 上。由于 $\boldsymbol\chi$ 无穷维，取可计算的观测函数
$$\Phi(x,\boldsymbol u_w)=(\Psi(x),\boldsymbol u_w(0)),\qquad N_\phi=n_\psi+m+n,\ n_\psi>n,$$
即只保留状态提升 $\Psi(x)=(\psi_1,\dots,\psi_{n_\psi})$ 与当前时刻的控制/扰动 $(u,w_o)$，$\psi_i$ 可取基函数或神经网络。数据 $\{(u_i,\hat w_{o,i},x_i,x_i^+)\}_{i=1}^M$（$\hat w_{o,i}$ 为扰动估计，可由非线性估计器或 Koopman 型估计器给出；数据独立采样自分布 $\mu$ 或在 $\mathcal{U}\times\mathcal{X}$ 中遍历，Assumption 1）上解正则化最小二乘
$$\min_{[\mathcal{K}_{N_\phi}]_{1:n_\psi}}\sum_{i=1}^M\|[\mathcal{K}_{N_\phi}]_{1:n_\psi}(\Psi(x_i),\hat u_{w,i})-\Psi(x_i^+)\|^2+\alpha\|[\mathcal{K}_{N_\phi}]_{1:n_\psi}\|_F^2,$$
记 $[\mathcal{K}_{N_\phi}]_{1:n_\psi}=[A\ B\ D]$，再用 $\min_C\sum_i\|C\Psi(x_i)-x_i\|^2+\beta\|C\|_F^2$ 求重构矩阵 $C$，得到线性 Koopman 预测器 $\hat s^+=A\hat s+B\hat u$、$\hat x=C\hat s$，并在此模型上陈述名义 KMPC 问题。作者随即给出本节的论证转折：由于所选观测函数 (2.5) **不构成 $\mathcal{F}_e$ 的正交基**，即使 $w_o$ 可测、采样遍历，也不能保证 $\mathcal{K}_{N_\phi}\to\mathcal{K}$（$N_\phi,M\to\infty$）；加上 $\hat w_o$ 的估计误差与 $\alpha,\beta\ne0$ 的正则化，建模误差不可避免——这直接引出鲁棒化的必要性。

## §3 r-KMPC 的构造链：含误差模型 → 逆映射与有界性 → 平衡点处理 → 管式控制器

**含不确定性的等价 Koopman 模型**。令 $s=\Psi(x)$，把全部不确定性归并为
$$s^+=As+Bu+\bar w(s,u,w_o,\hat w_o),\qquad x=Cs+v(s),$$
其中 $\bar w=D\hat w_o+w\in\bar{\mathcal{W}}=D\hat{\mathcal{W}}_o\oplus\mathcal{W}$、$v\in\mathcal{V}$，$\mathcal{W},\mathcal{V}$ 为含原点的凸集。对 $\Psi$ 施加 Assumption 2（Lipschitz 连续、$\psi_i$ 线性无关），并用广义梯度（Clarke 意义）的极大秩条件给出逆映射存在性引理：$\mathrm{rank}(\nabla\Psi(x))=n$ 时存在 Lipschitz 的 $\Psi^{-1}:\mathcal{S}_\Psi\to\mathcal{X}$。命题 2 以高斯核 $\psi_i(x)=e^{-\|x-c_i\|^2}$ 为例给出可操作性：任取 $n+1$ 个基函数，梯度 $\partial\bar\psi(x)=-2\,\mathrm{diag}\{\bar\psi(x)\}[(x-c_1)\cdots(x-c_{n+1})]^T$ 在 $c_1,\dots,c_{n+1}$ 线性无关时满秩 $n$（最坏测试点 $x=c_i$），故 $n_\psi$ 个核可给出 $\sum_{i=n}^{n_\psi}\frac{n_\psi!}{(n_\psi-i)!i!}$ 种 $\Psi^{-1}$ 选择；实用捷径是取 $\Psi(x)=(x,\bar\Psi(x))$，则 $\Psi^{-1}(s)=[I_n\ 0]s$。**有界性命题**（本文理论的关键支点）：$\mathcal{S}_\Psi$ 有界时 $\bar{\mathcal{W}},\mathcal{V}$ 有界，证明走 Lipschitz 链 $\|w\|\le L_s\|s\|+L_u\|u\|+L_{\delta w}\|\delta w_o\|+L_{\hat w}\|\hat w_o\|<\infty$ 与 $\mathcal{V}=\mathcal{X}\ominus C\mathcal{S}_\Psi$。原文强调**只需有界性、不需要收敛性**——这正是可以用低阶截断模型的理由；且 $\mathcal{W},\mathcal{V}$ 的大小受 Lipschitz 常数控制，故可在建模阶段额外最小化 $L_s,L_u,L_{\hat w}$（修改版问题 (3.4)：$V_{\mathcal{K}}+\alpha_sL_s+\alpha_uL_u+\alpha_wL_{\hat w}$，附数据对上的 Lipschitz 约束）以收紧不确定性集。

**平衡点的对齐**。$w_o=0$ 时 $(u,s,x)=(0,s_r,0)$（$s_r=\Psi(0)$）是 (3.1) 的平衡 iff $(w(s_r,0,0,0),v(s_r))=\bar As_r$，$\bar A=[I-A^T\ C^T]^T$。命题 3：该不确定性为零 iff $s_r\subseteq\mathrm{Ker}\,\bar A$ 或 $s_r=0$；(A,C) 可观测时化为 $s_r=0$。remark 讨论两条路线：强加 $As_r=s_r,Cs_r=0$（但由 PBH 检验，$A$ 有特征值 1 时模型不可观测）；或（Assumption 3）直接构造 $\Psi(0)=0$——用坐标平移即可实现，如 $\psi_i(x)=\psi_i'(x)-\psi_i'(0)$，且命题 2 的梯度论证在平移后仍成立。此后 (2.7) 成为 (3.1) 的名义模型。

**稳定性/可观测性的模型层性质**。命题 4（局部可镇定）：$\Psi(0)=0$ 下 (3.1)（$w_o=0$）在 $\mathcal{S}_\Psi\times\mathcal{U}$ 上可镇定 **iff** $f(x,u)$ 在 $\mathcal{X}\times\mathcal{U}$ 上可镇定（证明用 (3.1) 与原系统等价 + $\Psi(0)=0$）。命题 5（局部可观测）：若原系统复合映射对 $x_0$ 的 Jacobi 满秩且 $\Psi$ 单射，则 (3.1) 局部可观测。随后一个反例 remark 说明了为何还需 Assumption 4（$(A,B)$ 可镇定、$(A,C)$ 可观测）：即使 (3.1) 可镇定可观测，其**名义模型** (2.7) 可能两者皆不——例：$s^+=\mathrm{diag}\{1.01,1\}s+[0\ 1]^Tu+w$、$x=[0\ 1]s+v$，$w=-0.02s$、$v=[0.01\ 0]s$，系统本身可镇定可观测而名义模型不可。Assumption 4 可在解出 $A,B,C$ 后用可镇定性/可观测性矩阵直接检验。

**管式控制器**。控制律 $u=\hat u+K(s-\hat s)$，$F=A+BK$ Schur 稳定；误差动态 $e_s^+=Fe_s+\bar w$、$e_x=Ce_s+v$。取 $e_s$ 的鲁棒正不变集 $\mathcal{Z}_s\subseteq F\mathcal{Z}_s\oplus\bar{\mathcal{W}}$，则 $e_x\in C\mathcal{Z}_s\oplus\mathcal{V}:=\mathcal{Z}_x$。在线问题：名义 QP，代价 $\sum_{i=0}^{N-1}(\|\hat x_{k+i}\|_Q^2+\|\hat u_{k+i}\|_R^2)+V_f(\hat s_{k+N})$，终端代价 $V_f=\hat s^TP\hat s$、$P$ 解 Lyapunov 方程 $F^TPF-P=-(\bar Q+K^TRK)$（$\bar Q=C^TQC$）；约束为更紧的状态/控制集 $\mathcal{S}=\{\hat s\mid C\hat s\in\mathcal{X}\ominus\mathcal{Z}_x\}$、$\hat{\mathcal{U}}=\mathcal{U}\ominus K\mathcal{Z}_s$，初始约束 $s_k-\hat s_k\in\mathcal{Z}_s$、终端集 $\mathcal{S}_f$（$F\mathcal{S}_f\subseteq\mathcal{S}_f$ 的正不变集）。实际施加控制 $u_k=\hat u^*_{k|k}+K(s_k-\hat s_{k|k})$。remark 指出与经典线性 tube MPC 的本质差异：$\hat u$ 与 $K(\Psi(x)-\hat s)$ 对原状态 $x$ 都是**非线性**控制律——"线性鲁棒 MPC 设计产生非线性 MPC 律"（摘要宣称的贡献一即由此实现）。

**三条定理**。定理 1（递归可行性）：在假设 1–5 下初时可解则永远可行——证明构造下一时刻的候选次优解 $\hat s_{k+1|k+1}=\hat s_{k+1|k}$、$\bm{\hat u}^s=(\hat u^*_{k+1:k+N-1|k},K\hat s_{k+N|k})$，靠管继承约束。定理 2（闭环鲁棒性）：(a) 名义提升系统渐近收敛到原点（证 $V^*_{k+1}-V^*_k\le-\|\hat x_k\|_Q^2-\|\hat u_k\|_R^2$ 的单调性 + $Q,R\succ0$）；(b) 真实轨迹进管：$s_k\to\mathcal{Z}_s$、$u_k\to K\mathcal{Z}_s$、$x_k\to\mathcal{Z}_x$。remark 给出替代方案：代价改罚 $\|\hat s\|_{\tilde Q}^2$ 则不需要 (A,C) 可观测。定理 3（逐点收敛，$w_o=0$）：若 $E=(L_sI+L_uK^TK)\|\sum_{k=0}^{+\infty}F^k\|$ Schur 稳定，则 $x_k,u_k,s_k\to0$——证明把 $s^+=Fs+w$、$\|w\|\le\|s\|_{L_sI+L_uK^TK}$ 写成两个冗余互联系统，用小增益定理论证（$\bar E=\begin{bmatrix}0&E\\E&0\end{bmatrix}$ Schur 稳定）。

## 附录：不确定性集的统计学习估计

$\bar{\mathcal{W}},\mathcal{V}$ 依赖 $\Psi$ 与数据集，闭式不可得。附录用 Hoeffding (1963) 界：定义指示损失 $\ell_\star\in\{0,1\}$（样本的 $\bar w_i=\Psi(x_i^+)-\hat s_i^+$ 或 $v_i=x_i-C\Psi(x_i)$ 是否落在候选集内），经验风险 $\hat G_\star=\frac1L\sum\ell_\star$，检验 $\bar G_\star\ge\hat G_\star+\epsilon_\star$（置信 $1-\delta_r$，$\epsilon_\star=\sqrt{-\log(0.5\delta_r)/(2L)}$）；不满足则增大 $L$ 或扩大候选集，最后乘安全因子 $\gamma_w,\gamma_v>1$（算法 2）。命题 6 给出外扩的合理性：真集包含于候选集 $\oplus\Delta\mathcal{W}$，$\Delta\mathcal{W}=\{w\mid\|w\|\le L_sL_\Psi d_x+L_ud_u+L_{\delta w}d_{\delta w}+L_{\hat w}d_{\hat w}\}$、$\Delta\mathcal{V}=\{v\mid\|v\|\le L_vd_x\}$，$d_x,d_u,d_{\hat w}$ 为数据邻域最大距离。

## §4 仿真验证（具体参数与数字）

**Van der Pol 振子**（$\dot x_2=2x_2-10x_1^2x_2-0.8x_1-u$，$\|w_o\|_\infty\le0.4$，初始用 $w_o=0.4\sin(10\pi t)$）：$T=0.01$ s、$M=8\times10^5$（UDR 控制策略采数据）、thinplate 基 $\psi_i(x)=\|x-c_i\|^2\log\|x-c_i\|$、$n_\psi=4$；$\tilde Q=\mathrm{diag}\{1,1,0.1,0.1\}$、$R=0.1$、$N=10$；$K=[6.5\ 4.3\ -0.4\ 0.1]$（闭环极点 $\{0.91,0.93,0.98\pm0.003i\}$）。结果：正弦扰动下 $x,u$ 保持在以 $\hat x,\hat u$ 为中心的管内（$x\in\hat x\oplus\mathcal{Z}_s$，注：原文一处把 $u$ 的管写作 $K\mathcal{Z}_x$，与理论 $K\mathcal{Z}_s$ 不一致，见疑点）；无扰动时渐近收敛。与 KMPC 对比：400 步内 KMPC 不收敛而 r-KMPC 收敛；累计代价表（$J=\sum_{k=1}^{400}$，thinplate 核标称情形）：r-KMPC（$n_\psi=4$）258 vs KMPC $n_\psi=4/12/22$ 的 431/401/360——r-KMPC 用更少的观测函数得到更低代价；多项式核情形同型（247 vs 427/402）。还报告了一个预测器细节实验：$\Psi(0)\ne0$ 的 KMPC 预测器在 50000 个初始条件上的单步预测累计平方误差 56.4，与重置程序（$\Psi(0)=0$）的 55.9 相当——即平衡点对齐几乎不损失预测精度。

**倒立摆角度调节**（$\dot x_2=4g\sin x_1-3u\cos x_1$，$w_o=2\sin(10\pi t)$，$|x_1|\le1$ rad、$|x_2|\le2$ rad/s、$|u|\le20$）：$T=0.005$ s、$M=5\times10^4$、高斯核 $n_\psi=5$、$\tilde Q=I_5$、$R=0.1$、$N=10$、$x_0=(0.2,1)$。结果同型：管约束满足、名义收敛、无扰动渐近收敛；KMPC 中状态与控制呈发散趋势。代价表（高斯核标称）：r-KMPC 175 vs KMPC 434/228/203（$n_\psi=5/15/25$）；S-noise 333 vs 695/407/367；UDR-noise 191 vs 570/235/189（KMPC $n_\psi=25$ 时 189 略优）；SW-noise 201 vs 763/255/214。InvQuad 核情形同型。

**非仿射系统**（$\dot x_2=x_1^2+0.15u^3+0.1(1+x_2^2)u+\sin(0.1u)$，$w_o=\sin(10\pi t)$，$|x|\le2.5$、$|u|\le25$）：$T=0.005$ s、$M=5\times10^4$、polyharmonic 核 $\psi_i(x)=\|x-c_i\|\log\|x-c_i\|$、$n_\psi=5$、$\tilde Q=I_5$、$R=0.1$、$N=30$、$x_0=(0.6,-1.2)$。此例展示方法对控制输入非线性进入动力学（$u^3,\sin u$）的非仿射系统的适用性，结果同样是管满足 + 名义收敛 + 无扰动渐近收敛（无代价对比表）。

## 推理重建（标明为推断）

- "不假设 $\mathcal{K}_{N_\phi}\to\mathcal{K}$"的技术含义：定理只需 $\bar{\mathcal{W}},\mathcal{V}$ 有界（命题 1），而有界性只依赖 $\mathcal{S}_\Psi$ 有界与 Lipschitz 常数有限——因此截断阶数 $n_\psi$ 可保持很小（仿真中 $n_\psi=4,5$）；这是把"逼近不收敛"转化为"有界误差集"的论证桥，原文通过命题 1 前后的段落明说，链条由我串联。
- 定理 3 的小增益证明中把误差系统写成两个冗余互联系统是形式化技巧（$s_1^+=Fs_1+w_2$、$s_2^+=Fs_2+w_1$），其作用是把 $\|w\|\le\|s\|_{L_sI+L_uK^TK}$ 的增益条件套进标准小增益判据；这一构造的动机原文未解释，属推断。

## 限定条件与边界（原文显式）

- 结论以假设 1–5 为前提：数据分布/遍历、$\Psi$ Lipschitz 且线性无关、$\Psi(0)=0$、$(A,B)$ 可镇定 $(A,C)$ 可观测、紧缩集含原点于内部。原文指出这些可用"合适的 $\Psi$ 与足够的样本数 $M$"达成，但这是设计义务而非自动成立。
- 扰动须有界（$\mathcal{W}_o$ 紧、含原点）；$\hat w_o$ 须落在可计算的有界集 $\hat{\mathcal{W}}_o$ 内。
- 闭环鲁棒性是"进管"而非收敛；真正的渐近收敛只在 $w_o=0$ 且小增益条件 (3.13) 成立时得到。
- 双线性/开关模型作为替代建模路线被明确留作未来工作；讨论的结论段未宣称对实验数据的验证（三例均为仿真）。

## 疑点

- 正文组织段落写 "In Section 2 the main idea ... are obtained"，按内容应为 Section 3，疑原文笔误。
- 图 3/图 6 caption 中写 "$x\in\hat x\oplus\mathcal{Z}_s$ and $u\in\hat u\oplus K\mathcal{Z}_x$"，而定理 2(b) 为 $u\to K\mathcal{Z}_s$：两处 $\mathcal{Z}_x/\mathcal{Z}_s$ 不一致，疑为原文笔误。
- §2 开头小节标题层级在抽文中出现重复的 "## [section] Robust Koopman MPC"（源文件中存在重复标题块），按内容以第一次出现为准。
- 所有 `\cite` 编号丢失（KMPC 原始文献 Korda & Mezić、tube MPC 文献、扩展 Koopman 算子方案出处等均无法核对）；文中 "see e.g.."、"in," 等悬空引用多处。
- 三例仿真中 $\tilde Q$ 的记号：§3 定理 2 的 remark 提到可改罚 $\|\hat s\|_{\tilde Q}^2$，仿真节实际使用的即 $\tilde Q$（$\mathrm{diag}\{1,1,0.1,0.1\}$ 或 $I_5$），与正文 $Q$（输出罚）的关系未逐字说明，读表时需注意。
