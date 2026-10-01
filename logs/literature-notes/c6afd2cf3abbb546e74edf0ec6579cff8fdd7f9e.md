# 深度变分信息瓶颈用于不完整多组学数据整合（DeepIMV）

原题：A Variational Information Bottleneck Approach to Multi-Omics Data Integration（Lee, van der Schaar，AISTATS 2021，arXiv:2102.03014）。LaTeX 源，定位用章节/公式号。

## 问题定义：把多组学整合写成"不完整多视图问题"

论文先把多组学（omics）整合形式化为 Definition 1（Incomplete Multi-View Problem）：每个 omics 层是一个视图，$X_v\in\Xc_v$ 为第 $v$ 视图特征，视图缺失记 $x_v=\varnothing$；观测视图集合 $\Vc=\{v:x_v\neq\varnothing\}\subseteq[V]$ 称为 view-missing pattern，任务是对任意缺失模式的新样本做 $C$ 分类（$\Yc=[C]$）或回归（$\Yc=\mathbb{R}$）。动机来自 TCGA 等平台的现实：7,295 个癌症样本中有 3,282 个缺视图，且丢弃缺视图样本会大幅缩样本量、均值填补会扭曲边际与联合分布（§Introduction）。作者区分两条挑战：学公共空间中兼顾边际（marginal）与联合（joint）两方面的表征；在统一框架内灵活整合任意缺失模式。现有方法被批评为：CCA 系只适完备视图；不完整视图的生成式/矩阵分解法是纯无监督的，"重建相关的信息保住了，任务相关信息可能丢掉"；CPM-Nets 最接近但依赖重构定位潜变量、随机初始化导致训练样本潜表示之间无内在关系、必须同时更新全部样本（内存负担），且只适分类。

## 四组件架构与双重 IB 损失

DeepIMV 由四组网络构成：$V$ 个视图专属编码器（$\theta_v$，随机映射到公共潜空间 $\Zc$）、一个 PoE 模块、一个多视图预测器（$\psi$）、$V$ 个视图专属预测器（$\phi_v$）。两个 IB 目标都从 DVIB 变分化来：

联合表征 $Z$（由编码块 $q_\theta(Z|\bar{\Xv})$ 组合各视图编码器输出）上施加

$$\mathcal{L}_{\text{IB-J}}^{\theta,\psi}=-I(Y;Z)+\beta I(\bar{\Xv};Z)\approx\mathbb{E}\big[-\log q_\psi(y|z)\big]+\beta\,\mathbb{E}\big[KL\big(q_\theta(Z|\bar{\xv})\,\|\,q(Z)\big)\big]$$

边际表征 $Z_v$ 上对每个视图施加 $\mathcal{L}_{\text{IB-M}}^{\theta_v,\phi_v}=-I(Y_v;Z_v)+\beta_v I(X_v;Z_v)$，同样变分化。总损失 $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{IB-J}}+\alpha\sum_{v\in\Vc}\mathcal{L}_{\text{IB-M}}$（Eq.(4)）。作者对边际 IB 的论证是：最小化它使 $Z_v$ 成为 $X_v$ 对 $Y_v$ 的最小充分统计量，迫使各编码器发展"视图专属专长"（view-specific expertise），从而缓解 PoE 难训练的问题——原文承认训练 PoE 本来"困难，需要对观测视图人为子采样或用对比散度变体"。

## PoE 因子化：为什么是乘不是混合

关键设计决策在 §4.2：联合后验因子化为边际后验之积 $p(z|\bar{\xv})\approx C\cdot p(z)\prod_{v\in\Vc}q_{\theta_v}(z|x_v)\triangleq q_\theta(z|\bar{\xv})$。与 MoE（$q_\theta(z|x_1,\cdots,x_V)=\sum_v\alpha_v q_{\theta_v}(z|x_v)$）对比给出两条理由：其一，PoE 可在求联合表征时直接跳过缺失视图，故任意缺失模式都能用，无需填补或辅助推断；其二，PoE（联合后验）能产出比单个专家（边际后验）**更尖锐**的分布，使各专家可专注于目标任务的特定侧面，这对"各视图常含互补信息或信息量不均"的多组学场景是合意性质。高斯情形闭式可解：$q_{\theta_v}(z|x_v)=\Norm(z|\mu_v,\Sigma_v)$、先验 $p(z)=\Norm(z|\mu_0,\Sigma_0)$ 时，$q_\theta(z|\bar{\xv})=\Norm(z|\mu,\Sigma)$，$\Sigma=(\Sigma_0^{-1}+\sum_{v\in\Vc}\Sigma_v^{-1})^{-1}$，$\mu=(\mu_0\Sigma_0^{-1}+\sum_{v\in\Vc}\mu_v\Sigma_v^{-1})\Sigma$。推断：正是这个闭式乘积结构（对数域为加法）使"缺失视图自动不出现在乘积里"，PoE 对不完整视图的灵活性由此而来——原文明说第一条，第二条（尖锐性）引前人结论。

## TCGA 一年死亡率预测

数据：4 个视图（mRNA、DNA 甲基化、microRNA、反相蛋白阵列），3 层 omics，7,295 样本中 3,282 缺视图（各视图缺失率 0.10/0.24/0.13/0.74）。所有方法在每视图先用核 PCA（多项式核）取 100 维特征。对比 2 基线（Base1 拼接、Base2 集成）与 GCCA/DCCA/DCCAE/MVAE/CPM-Nets/MOFA；不能处理不完整视图的方法用均值填补。结果以 AUROC（mean $\pm$ 95% CI，10 次随机 64/16/20 划分）报告，训练时把 $N_I=3,282$ 个缺视图样本加到 $N_C=3,210$ 个完备样本上，测试时人为制造 $|\Vc^n|=1\ldots4$ 的缺失（Table 3）：
- 4 Views/incomplete：DeepIMV 0.801$\pm$0.01，超过 Base2 0.790、GCCA 0.792、CPM-Nets 0.788；3 Views/incomplete 0.791$\pm$0.01 也全面领先。
- 即便只用完备样本训练，DeepIMV 在 2/3/4 视图的不完整测试中也最高（除 1 View，1 View/incomplete 最高是 DeepIMV 0.724$\pm$0.02 本身）。
- 换用 MVAE 高级填补（$|\Vc^n|=3$）：DeepIMV 0.791$\pm$0.01 仍最高，超过 MVAE 填补下的 DCCA/Base2（0.784）与 Base1（0.771）。
- 消融（Table 4）：MoE 0.768 → MoE+marginal IBs 0.790；PoE 0.783 → PoE+marginal IBs 0.801（4 Views）。即 PoE 与边际 IB 各自贡献增益、叠加最佳。
- 信息量探针（Table 5）：$I(Y_1;Z_1)=0.319$、$I(Y_2;Z_2)=0.506$、$I(Y_3;Z_3)=0.487$、$I(Y_4;Z_4)=0.157$、$I(Y;Z)=0.562$。结合 PCA 可视化：加入信息量最高的 View 2 后样本向类边界反向移动；不含信息量最低的 View 4（$\Vc=\{1,2,3\}$）时潜表征与 4 视图几乎相同。作者据此提出应用：用 $I(Y_v;Z_v)$ 指导"哪些 omics 层不必测"，节省实验成本。

## CCLE 药物敏感性：缺失率鲁棒性

数据：6 视图 5 层 omics（DNA 拷贝数、甲基化、mRNA、microRNA、RPPA、代谢物），504 细胞系，4 种药物（Irinotecan、Panobinostat、Lapatinib、PLX4720），按 ActArea 四分位把顶部 25% 标"敏感"。人为构造缺失：随机选 $N_I=N\times R$ 个样本、从 $2^V-2$ 种缺失模式中选（100 次随机划分）。两条曲线实验（Fig. 3）：视图数从 $\{1,2\}$ 增到 $\{1,\ldots,6\}$（$R=0.6$）时 DeepIMV 在所有药物上（多数时候）持续提升，DCCA/DCCAE 因最多只用两视图而饱和，GCCA 随视图单调升；缺失率 $R$ 从 0 到 1（$V=6$）时 DeepIMV 在 Irinotecan 和 Panobinostat 上胜过所有基准，Lapatinib/PLX4720 上与最佳基准相当且"其他方法常失败时最鲁棒"。Panobinostat（$M=6,R=0.6$）的填补对比（Table 6）：DeepIMV 0.768$\pm$0.01，超过 MVAE 填补下的 Base1 0.758、Base2 0.752。

## 边界与疑点

- 原文明说的边界：结论依赖核 PCA 预降维（为避"维度灾难"），MOFA 例外直接用稀疏因子；未来工作承认需在 omics 数据中引入稀疏性以应对高维（§Conclusion）。
- 分析重建：注释文本指出修正后的 CPM-Nets 在回归任务上"所有预测收敛到真标签均值"而退出对比——这是负面结果，原文以脚注/注释形式记录，说明 CPM-Nets 的聚类友好表征不适回归。
- 疑点：Table 3 中 1 View/complete 列 Base2 最高（0.711）而 DeepIMV 为 0.701，即单视图时多视图方法无优势，原文只在行文里顺带承认（"except for 1 View"）。超参 $(\alpha,\beta)$ 敏感性分析放在补充材料，正文未给数值。
- 源文本含大量被注释掉的旧段落（`%` 开头），如 MFM 对比草稿，已不作为论文主张依据。
