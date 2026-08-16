# CM-LLM 块稀疏时空因果方法理论提升

本文档给出当前改进方案的数学依据、复杂度结论、适用边界和下一阶段可实施方向。所有块公式统一使用 `$$...$$`，以兼容常见 Markdown 数学渲染器。

## 1. 问题定义

设配电系统部署 $N$ 个传感器，每个传感器记录 $F$ 个不同物理变量：

$$
x_i(t)=
\begin{bmatrix}
x_{i1}(t)&\cdots&x_{iF}(t)
\end{bmatrix}^{\top}
\in\mathbb R^F.
$$

完整系统状态为：

$$
X(t)=
\begin{bmatrix}
x_1(t)^{\top}&\cdots&x_N(t)^{\top}
\end{bmatrix}^{\top}
\in\mathbb R^{NF}.
$$

目标是在保留变量级关系的同时，使动态关系发现、图传播和大模型重构可以扩展到数百乃至数千传感器。

## 2. 特征平均为什么必然损失信息

### 2.1 线性代数证明

旧式平均等价于：

$$
y_i(t)=a^{\top}x_i(t),
\qquad
a=\frac{1}{F}\mathbf 1.
$$

映射 $a^{\top}:\mathbb R^F\rightarrow\mathbb R$ 的秩为 1，因此零空间维数为：

$$
\dim\ker(a^{\top})=F-1.
$$

任意满足：

$$
a^{\top}v=0
$$

的变化方向 $v$ 在平均值中完全不可见。例如电压分量增加、潮流分量减少，只要投影后抵消，平均图就无法区分这些物理上完全不同的状态。

若真实目标关系为：

$$
r(t)=c^{\top}x_i(t)+\epsilon(t),
$$

只有当 $c$ 与 $a$ 共线时，才存在标量 $\gamma$ 使：

$$
c^{\top}x_i=\gamma a^{\top}x_i
$$

对所有 $x_i$ 成立。一般的电压-功率、功率-潮流交叉关系不满足该条件，所以平均值不是充分统计量。

### 2.2 信息论证明

$y_i=a^{\top}x_i$ 是 $x_i$ 的确定性函数。由数据处理不等式：

$$
I(y_i;x_j)\le I(x_i;x_j).
$$

等号只在 $y_i$ 对目标关系保持充分时成立。对多种物理变量的秩一投影，通常为严格不等式。逐变量标准化可以统一尺度，但不会提高投影秩，也无法恢复已丢失的信息。

### 2.3 预测风险分解

设最优完整信息预测器为：

$$
f^{\star}(x_i)=\mathbb E[r\mid x_i],
$$

仅使用平均值的最优预测器为：

$$
g^{\star}(y_i)=\mathbb E[r\mid y_i].
$$

平方损失下，两者的 Bayes 风险差为：

$$
\mathcal R(g^{\star})-\mathcal R(f^{\star})
=\mathbb E\left[
\operatorname{Var}
\left(
\mathbb E[r\mid x_i]\mid y_i
\right)
\right]\ge0.
$$

只要平均值相同的状态仍对应不同的条件期望，风险差就严格大于零。这给出了变量平均降低精度的直接统计结论。

## 3. 完全展开的维度爆炸

若将每个 `(sensor,feature)` 作为独立节点，节点数为 $NF$。考虑 $p$ 个滞后时，稠密动态关系数为：

$$
p(NF)^2=N^2pF^2.
$$

存储、回归设计矩阵和图神经网络消息都可能达到二次传感器复杂度。PC 类条件独立方法还会枚举条件集合，最坏复杂度随最大节点度指数增长，因此不适合大规模馈线。

## 4. 块动态因果图

保留传感器作为拓扑节点，但让节点承载完整向量 $x_i(t)$。边 $i\rightarrow j$ 在滞后 $\tau$ 上保存：

$$
B_{ij}^{(\tau)}\in\mathbb R^{F\times F}.
$$

动态结构方程为：

$$
x_j(t)=
\sum_{i\in\mathcal C_j\cup\{j\}}
\sum_{\tau=1}^{p}
\left(B_{ij}^{(\tau)}\right)^{\top}x_i(t-\tau)
+\epsilon_j(t).
$$

分量形式为：

$$
x_{j,b}(t)=
\sum_i\sum_{\tau=1}^{p}\sum_{a=1}^{F}
B_{ij,ab}^{(\tau)}x_{i,a}(t-\tau)
+\epsilon_{j,b}(t).
$$

该表示与展开变量图具有相同的变量对表达能力，但只对选中的传感器边分配 $F\times F$ 块，不创建全局稠密 $(NF)\times(NF)$ 矩阵。

## 5. 物理候选约束

### 5.1 电气距离

支路阻抗距离：

$$
d_e(i,j)=
\sum_{(u,v)\in path(i,j)}
\sqrt{r_{uv}^2+x_{uv}^2}.
$$

对每个目标传感器只保留距离最小的 $d$ 个候选传感器，并强制保留已知物理邻边：

$$
|\mathcal C_j|\le d
$$

在物理度数不超过 $d$ 时成立。相比纯拓扑跳数，阻抗距离更接近配电网电压和潮流灵敏度。

### 5.2 变量模板

定义物理允许矩阵：

$$
M_{ij}\in\{0,1\}^{F\times F}.
$$

有效系数为：

$$
\widetilde B_{ij}^{(\tau)}
=M_{ij}\odot B_{ij}^{(\tau)}.
$$

对恒功率注入假设，远端电压和潮流不应指向有功或无功注入；对电压、相角和支路潮流目标，允许注入和上游状态解释。该模板排除明显不合建模假设的关系，但未知变量类型保持开放，避免不可靠的过强硬约束。

### 5.3 候选筛选不等于因果证明

电气邻域提供结构先验，滞后提供时间方向，但它们不能消除未观测混杂。因此统计边应解释为：

$$
x_i(t-\tau)\text{ 对 }x_j(t)
\text{ 的条件滞后预测关系},
$$

而不是无条件的干预因果效应。物理拓扑边与统计发现边必须在报告中分开标注。

## 6. Sparse-group VAR 推导

对目标传感器 $j$，把候选源和全部滞后拼成设计矩阵 $X_j$，响应为 $Y_j$。求解：

$$
\min_B f(B)+g(B),
$$

其中：

$$
f(B)=\frac{1}{2n}\lVert Y_j-X_jB\rVert_F^2,
$$

$$
g(B)=
\lambda_g\sum_i\omega_{ij}\lVert B_{ij}\rVert_F
+\lambda_1\lVert B\rVert_1.
$$

组惩罚产生传感器边级稀疏，L1 产生变量对级稀疏。物理边取较小 $\omega_{ij}$，自身块取 $\omega_{jj}=0$。

光滑项梯度：

$$
\nabla f(B)=\frac{1}{n}X_j^{\top}(X_jB-Y_j).
$$

梯度 Lipschitz 常数：

$$
L=\frac{\lVert X_j\rVert_2^2}{n}.
$$

取 $\eta=1/L$，梯度步为：

$$
U=B^{(k)}-\eta\nabla f(B^{(k)}).
$$

元素软阈值：

$$
\bar U=
\operatorname{sign}(U)
\max(|U|-\eta\lambda_1,0).
$$

块收缩：

$$
B_{ij}^{(k+1)}=
\left(
1-\frac{\eta\lambda_g\omega_{ij}}
{\max(\lVert\bar U_{ij}\rVert_F,10^{-12})}
\right)_+\bar U_{ij}.
$$

物理模板在块收缩前施加，保证禁用变量关系保持为零。不同目标传感器的目标函数互不共享参数，可以并行求解。

## 7. 可扩展性结论

设每个目标最多有 $d$ 个空间候选和一个自身块，每个块包含 $pF^2$ 个参数。存储复杂度：

$$
O(N(d+1)pF^2).
$$

每次完整梯度迭代复杂度：

$$
O(TN(d+1)pF^2).
$$

当 $d,p,F$ 固定时，两者都关于 $N$ 线性增长。相比稠密展开，理论缩减比例为：

$$
\frac{N^2pF^2}{N(d+1)pF^2}
=\frac{N}{d+1}.
$$

对 $N=1000,d=4$，理论候选规模减少 200 倍。当前 100 传感器烟雾实验保存 35,640 个块候选系数，而稠密展开需要 720,000 个，实际缩减 20.2 倍。

## 8. 稀疏关系 DGP

边块首先产生变量消息：

$$
m_{ij}(t)=
\sum_{\tau=1}^{p}
\left(B_{ij}^{(\tau)}\right)^{\top}z_i(t-\tau).
$$

祖先和后代隐藏消息分别为：

$$
a_j(t)=
\frac{\sum_{i:i\rightarrow j}w_{ij}h_i(t)}
{\sum_{i:i\rightarrow j}w_{ij}},
$$

$$
d_i(t)=
\frac{\sum_{j:i\rightarrow j}w_{ij}h_j(t)}
{\sum_{j:i\rightarrow j}w_{ij}}.
$$

第一层更新：

$$
h_i^{\prime}=\operatorname{LN}\left(
h_i+\operatorname{Dropout}
\left[
\operatorname{GELU}
\left(W_0h_i+W_aa_i+W_dd_i+W_rm_i\right)
\right]
\right).
$$

若实际边数为 $E$，隐藏传播复杂度为：

$$
O(BTEH),
$$

变量块消息复杂度为：

$$
O(BTEpF^2).
$$

两者都不依赖 $N^2$。

## 9. 结构残差正则

直接惩罚 $\lVert R_G(\widehat X)\rVert^2$ 会迫使预测比真实数据更贴合近似 VAR，可能引入模型偏差。改进方案定义：

$$
R_G(X)_j(t)=z_j(t)
-\sum_i\sum_{\tau=1}^{p}
\left(B_{ij}^{(\tau)}\right)^{\top}z_i(t-\tau),
$$

并惩罚：

$$
\mathcal L_G=
\left\lVert
R_G(\widehat X)-R_G(X)
\right\rVert_{2,valid}^2.
$$

由于 $R_G$ 是仿射标准化后的线性动态算子，令重构误差 $e=\widehat X-X$，可写成：

$$
R_G(\widehat X)-R_G(X)=\mathcal A_G(e),
$$

其中 $\mathcal A_G$ 是去除常数项后的线性部分。因此该项约束重构误差的结构传播，不惩罚真实序列自身不可解释的创新噪声。当 $\widehat X=X$ 时，该项严格为零。

从误差传播先验：

$$
\mathcal A_G(e)\sim\mathcal N(0,\sigma_G^2I)
$$

出发，其负对数先验与 $\mathcal L_G$ 成正比，因此总目标可以解释为最大后验估计。

## 10. LoRA 的参数隔离

对 Qwen 的任一被选择线性矩阵：

$$
W=W_0+\frac{\alpha}{r}BA,
$$

$$
A\in\mathbb R^{r\times k},
\qquad
B\in\mathbb R^{d_o\times r},
\qquad
r\ll\min(k,d_o).
$$

训练时：

$$
\nabla_{W_0}\mathcal L=0,
\qquad
\nabla_A\mathcal L\ne0,
\qquad
\nabla_B\mathcal L\ne0.
$$

因此 Qwen 原始能力保存在 $W_0$ 中，任务适配被隔离到低秩增量和外部时空模块。部署时可以独立加载、替换或关闭 adapter，不需要覆盖本地大模型权重。

## 11. 提示统计的进一步改进

当前提示词使用归一化观测值的全局均值、标准差和端点趋势。它不会替代主数值张量，因此不存在主通道信息损失，但文本摘要本身仍然较粗。

更严格的方案是逐变量统计：

$$
s_f=
\begin{bmatrix}
\mu_f&\sigma_f&\Delta_f&r_f^{obs}
\end{bmatrix},
$$

其中 $r_f^{obs}$ 为变量 $f$ 的观测比例。再用共享映射：

$$
e_f^{stat}=\phi_{stat}(s_f)+e_f^{feature}.
$$

这些统计 token 可放在数值 token 之前。复杂度只增加 $O(FH_q)$，不会随传感器数平方增长，也不会把不同物理量混成一个均值。

若进一步需要传感器级摘要，可采用可分解表示：

$$
e_{if}^{stat}=
\phi_s(s_{if})+e_i^{sensor}+e_f^{feature},
$$

产生 $NF$ 个短统计 token。它的序列开销是 $O(NF)$，仍远小于完整时间序列的 $O(TNF)$。

## 12. 单向 Qwen 对插补任务的限制与双向方案

当前 flatten 顺序是时间优先，Qwen 使用 causal attention。因此位置 $k$ 的输出只能依赖：

$$
\widehat x_k=f(x_1,\ldots,x_k;\text{prompt}),
$$

不能直接利用 $x_{k+1},\ldots,x_{TN}$。这对未来预测合理，但对中间缺失插补并非最优。

可采用共享基座和共享 LoRA 的双向两遍推理：

$$
H^{\rightarrow}=Q_{W_0,\Delta W}(E_1,\ldots,E_L),
$$

$$
H^{\leftarrow}=\operatorname{Reverse}
\left[
Q_{W_0,\Delta W}(E_L,\ldots,E_1)
\right].
$$

用门控融合：

$$
g_k=\sigma\left(W_g[H_k^{\rightarrow};H_k^{\leftarrow}]+b_g\right),
$$

$$
H_k=g_k\odot H_k^{\rightarrow}
+(1-g_k)\odot H_k^{\leftarrow}.
$$

同一组 $W_0$ 和 LoRA 参数在两个方向共享，因此参数量不翻倍，只将前向计算量近似增加一倍。预测任务继续只用正向分支，插补和超分辨率启用双向分支。

## 13. 更大系统的进一步降维

当变量种类 $F$ 也很大时，单个边块的 $pF^2$ 仍可能昂贵。可以对关系块做共享低秩分解：

$$
B_{ij}^{(\tau)}=
\sum_{r=1}^{R}
g_{ijr}^{(\tau)}U_rV_r^{\top},
\qquad R\ll F.
$$

其中 $U_r,V_r\in\mathbb R^{F}$ 是跨边共享的变量关系基，$g_{ijr}^{(\tau)}$ 是边特异系数。参数复杂度从：

$$
O(NdpF^2)
$$

降为：

$$
O(RF+NdpR).
$$

只要 $R\ll F$，该分解在保留变量交互的同时进一步控制维度。

## 14. 图稳定性选择

为减少单次时间序列拟合的偶然边，可将训练段划分为保留时间连续性的 block bootstrap 样本。设第 $b$ 次重采样得到边集合 $E^{(b)}$，边选择概率为：

$$
\widehat\pi_{ijab}^{(\tau)}=
\frac{1}{B_s}
\sum_{b=1}^{B_s}
\mathbf 1
\left(
|B_{ijab}^{(\tau,b)}|>\delta
\right).
$$

只保留：

$$
\widehat\pi_{ijab}^{(\tau)}\ge\pi_{min}
$$

的关系。目标传感器和 bootstrap 重采样都可并行，适合离线大系统图估计。

## 15. 物理灵敏度候选

电气距离只使用阻抗和路径，不区分当前运行点。更精细的候选权重可以来自潮流 Jacobian：

$$
\begin{bmatrix}
\Delta P\\
\Delta Q
\end{bmatrix}
=
J
\begin{bmatrix}
\Delta\theta\\
\Delta V
\end{bmatrix}.
$$

局部线性灵敏度为：

$$
\begin{bmatrix}
\Delta\theta\\
\Delta V
\end{bmatrix}
=J^{-1}
\begin{bmatrix}
\Delta P\\
\Delta Q
\end{bmatrix}.
$$

可按 $|\partial V_j/\partial P_i|$、$|\partial V_j/\partial Q_i|$ 和潮流灵敏度筛选候选变量对。为避免单运行点偏差，应在多个代表运行点计算并取分位数或稳定平均。

## 16. 实证结论与解释边界

在相同数据、轻量骨干、随机种子和 30 epoch 预算下，块图相对旧平均图：

| 任务 | MAE 降幅 |
|---|---:|
| 插补 | 12.6% |
| 预测 | 3.5% |
| 超分辨率 | 16.4% |

结果与第 2 节的风险分析一致：保留变量级关系提高了重构表达能力。结构残差正则进一步带来约 0.1% 至 0.8% 的测试 MAE 改善，但幅度小，应作为弱先验。

这些数值来自轻量骨干对照实验，不能替代 Qwen3.5-9B 的完整长训练结果。当前已经验证 Qwen 前向、冻结状态和 LoRA 梯度路径，但没有虚构未完成实验的最终精度。

## 17. 推荐实施顺序

后续若继续提升，建议按以下顺序实施：

1. 增加逐变量统计 token，消除提示词摘要层面的变量混合，代价最小。
2. 为插补和超分辨率增加共享 LoRA 的双向两遍推理，预测保持单向。
3. 加入 block bootstrap 稳定性选择，报告边选择概率而不只报告单次系数。
4. 用多运行点潮流 Jacobian 灵敏度替换或融合阻抗距离候选。
5. 当 $F$ 显著增长时，再引入共享低秩关系基；当前 $F=6$ 时没有必要增加该复杂度。
6. 对超大馈线做区域分解，仅在边界传感器交换稀疏消息，实现分布式图估计和训练。
