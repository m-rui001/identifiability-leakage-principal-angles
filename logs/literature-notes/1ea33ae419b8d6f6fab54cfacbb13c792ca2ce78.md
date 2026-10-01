# MIMO Wiener 型 Koopman 模型：放弃线性状态重构换取降阶能力的深度学习辨识

原题：Identification of MIMO Wiener-type Koopman Models for Data-Driven Model Reduction using Deep Learning（Schulze, Doncevic, Mitsos，Computers & Chemical Engineering，2022，arXiv:2201.12669）。LaTeX 源，定位用章节/公式号。

## 推导主线：从 Surana 的双线性 Koopman 到 Wiener 结构的关键一步是"可逆坐标变换替代线性重构"

论文的理论部分从 Surana 的输入仿射 Koopman 模型出发：对 $\dot{\bm x}=\bm f(\bm x)+\sum_i\bm h_i(\bm x)u_i$，若状态可由特征函数线性重构 $\bm x=C^{\varphi}\bm\varphi(\bm x)$，则 Koopman 动力学 $\ddt{\bm\varphi}=\Lambda\bm\varphi+\sum_i\nabla_x\bm\varphi^T\bm h_i(\bm x)u_i$ 中，$\nabla_x\bm\varphi^T\bm h_i(\bm x)$ 落在 $\bm\varphi$ 的张成内时得双线性 Koopman 模型 $\ddt{\bm z}=A\bm z+\sum_iB^{(i)}\bm zu_i$、$\bm x=C\bm z$；更强的条件 $\nabla_x\bm\varphi^T h_i(\bm x)\equiv\mathrm{const.}$ 下退化为线性模型 $\ddt{\bm z}=A\bm z+B\bm u$（与 DMD with controls 相关）。作者在此做出核心改换：**放弃线性状态重构假设**，改设坐标变换 $\bm\varphi(\bm x)$、$\bm T(\bm x)$ 连续可逆（连续双射），把 $\bm x=C^{\varphi}\bm\varphi$ 换成 $\bm x=\bm\varphi^{-1}(\bm\varphi)$，得
$$\ddt{\bm z}=A\bm z+\sum_{i}B^{(i)}\bm zu_i,\qquad\bm x=\bm T^{-1}(\bm z)$$
以及在 Condition 下化为
$$\ddt{\bm z}=A\bm z+B\bm u,\qquad\bm x=\bm T^{-1}(\bm z)$$
后者正是 MIMO Wiener 结构（线性动力块+静态非线性输出块）。作者指出两类模型形式上相似但矩阵与变换"可能根本不同"。Wiener 结构的理论边界由 Boyd–Chua 定理给出：只能任意精度逼近"衰减记忆"（fading memory）系统——当前状态只依赖近期输入，对应唯一稳态；作者认为这一性质比 Condition 的有限维可满足性更易判断。附带说明：给 Koopman 观测添加输入的非线性函数可扩展到 Hammerstein–Wiener；双线性形式仅在连续时间生成元下成立、离散时间一般不保（除非可分系统），故精细采样是离散双线性的前提。

## 辨识策略：Lusch 框架加输入项，损失函数三项

辨识（§Identification）用深度自编码器夹持离散 LTI 动力块：编码器 $\bm T$ 提供变换初值（可不属于控制模型），解码器 $\bm T^{\dagger}$ 沿轨迹恢复状态；$A$ 被限制为块对角以发现本征动力学。数据用足够细采样的阶跃响应快照（全状态、无噪声），切成 $p$ 步短轨迹分批。损失为三项 MSE 加权（均按数据均方和归一化为 NMSE）：
$$L_1=\frac{1}{p}\sum_{k=0}^{p}\|\bm x_k-\bm T^{\dagger}(\bm T(\bm x_k))\|_\mathrm{MSE},\quad L_2=\frac{1}{p-1}\sum_{k=0}^{p-1}\|\bm x_{k+1}-\bm T^{\dagger}(\bm z_{k+1}(\bm x_k))\|_\mathrm{MSE},\quad L_3=\frac{1}{p-1}\sum_{k=0}^{p-1}\|\bm x_{k+1}-\bm T^{\dagger}(\bm z_{k+1}(\bm x_0))\|_\mathrm{MSE}$$
即重构、沿轨迹单步预测、从初值出发的多步预测（应对慢动态），另加 $\ell_1$ 正则。该策略不依赖特殊输入信号、同时辨识所有模块；虽然变换块须可逆，但只要稳态唯一，仍可辨识各别输出有输入多重性的系统。基准对照（线性/双线性）用同一框架实现：去掉解码器隐层、保留非线性编码，得到的是 $\bm x=C\bm z$ 版本的离散化 Surana 模型；双线性模型额外含显式线性控制项（经验上利于训练）。评估只给初值与输入序列做全程前向仿真，以 NMSE 评判。

## 三个案例研究

**案例一：输入多重性系统**（$\dot{x}_1=-0.1x_1+u$，$\dot{x}_2=x_1^2-x_2$，$y=x_2$；$y^s=100u^2$ 有输入多重性但稳态唯一）：100 个随机阶跃训练、20 个测试（$t_\mathrm{step}=200/100$，$\Delta t=1$）。$n_z=2$（20 神经元单隐层）时：Wiener NMSE $2\times10^{-5}$、双线性 $2\times10^{-4}$、线性 0.15；线性模型升到 $n_z=10$ 反而恶化到 0.77。关键结果在 $n_z=1$：线性与双线性模型都失败，Wiener 模型仍成功预测动态与稳态行为——作者读作"该系统允许两状态的联合编码—解码（非线性投影），而线性重构不适用"，直接印证推导中放弃线性重构的必要性。

**案例二：放热反应器**（A→B，CSTR，输入为热负荷 $\dot{Q}$，$\dot{Q}=0$ 时三个稳态；目标为高收率操作区的**局部**模型）：500 个随机阶跃（$\dot{Q}\in[-2000,10{,}000]$ kJ/h，步长 4 h，采样 1/min）训练、15 个 2 h 阶跃测试。极简配置 $n_z=1$：Wiener 与双线性 NMSE 均为 0.002，线性 0.009（$n_z=10$ 仅降到 0.007）。区别在误差性质：双线性动态响应略优，但**只有 Wiener 模型在整个操作范围内稳态偏移可忽略**——线性模型的稳态偏移不随 $n_z$ 增大消失。作者据此论断：仅靠引入少量神经元的浅层 NN 解码器即可获得高精度模型，可用于需要更小安全裕度的高精度质量跟踪。

**案例三：高纯度甲醇-丙醇精馏塔**（8 块平衡板+冷凝器/再沸器，$n_x=10$，输入为轻组分进料组成 $u_1=x_f$ 与回流量 $u_2=L$；塔内呈非线性行波相干结构，故预期极低阶模型即可）：400 个随机阶跃（$x_f\in[0.5,0.6]$，$L\in[0.0155,0.0175]$ kmol/min，步长 2 h），塔板上摩尔分数跨数量级，按惯例对轻组分（提馏段）与重组分（精馏段）摩尔分数做对数变换训练（作者指出这本身即 Wiener 型建模的一种变体）。$n_z=2$（10 神经元隐层）：Wiener NMSE 0.0003、双线性 0.0012、线性 0.0025（$n_z=10$ 时 0.0013）——Wiener 以最低复杂度取得最优。塔底/塔顶组成预测误差与全塔同量级。

## 推理链重建与疑点

- 原文明说：所有案例采用对称自编码器（解码器层排布为编码器逆序）；MATLAB System Identification Toolbox 的 MIMO Wiener 辨识无法生成可接受精度的低阶模型，结果只放附录。
- 分析重建：Wiener 与双线性在案例二 NMSE 相同（0.002）但误差结构不同（稳态偏移 vs 动态偏差），原文以图示说明而未给出稳态误差的量化指标，"Wiener 稳态偏移可忽略"是从图形读取的定性判断。
- 疑点一：案例一的线性模型 $n_z=10$ 时 NMSE 0.77 比低阶时更差，作者只说"degrades the predictions rather than improving"，未解释（可能为过拟合或训练不收敛），属未决问题。
- 疑点二：案例三对数变换既是建模惯例又等价于输出非线性块，Wiener 的优势有多少来自学习到的解码器、多少来自预设的对数变换，未做消融。
- 疑点三：Abstract 强调"strongest model reduction capabilities"，但三案例中 Wiener 的 $n_z$ 与双线性相同（2/1/2），差异在解码器而非子空间维数——"更强降阶"实为"同维下更准/可更低阶而不崩"，表述略有夸大。
- 边界：假设无噪声全状态快照数据（阶跃响应即可）；fade memory/唯一稳态是结构前提，多稳态系统只能做局部模型（案例二明示）；双线性离散模型依赖细采样。
