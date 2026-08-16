# CM-LLM case33bw 时空因果复现

本项目复现论文 CM-LLM 的掩码重构、三模态融合、DGP、Qwen 和 LoRA 方法，并针对
大规模配电系统将原有标量因果图升级为物理约束的块稀疏动态因果图。

当前正式方案不会对传感器上的不同量测求平均。每个传感器保留完整向量

```text
[vm_pu, va_degree, p_inj_mw, q_inj_mvar, p_upstream_mw, q_upstream_mvar]
```

每条图边保存 `lag x source_feature x target_feature` 系数块。候选边由 MATPOWER
支路阻抗电气距离、物理拓扑和 `max_neighbors` 限制，统计估计采用 sparse-group VAR，
图传播使用稀疏 `edge_index`，不会创建稠密的 `(N*F) x (N*F)` 邻接矩阵。

详细内容：

- [整体技术文档](docs/technical_report.md)
- [理论提升文档](docs/theoretical_improvements.md)

## 已验证环境

MATLAB R2024b、MATPOWER 8.1、Python 3.10、PyTorch 2.10、Transformers 5.15、
PEFT 0.20、RTX 5090、本地 Qwen3.5-9B。

```powershell
$python = 'D:\Conda\envs\power_llm\python.exe'
```

## 数据与检查

```powershell
& $python scripts\generate_data.py --config configs\default.json
& $python scripts\preflight.py --config configs\default.json
& $python scripts\export_causal_graph.py --config configs\default.json
& $python scripts\scalability_smoke.py --sensors 100 --n-jobs 4
& $python -m unittest discover -s tests -v
```

数据生成器逐时刻调用本地 `D:\luosipeng\matpower8.1` 的 `runpf(case33bw)`，同时
输出 10 个传感器之间的阻抗加权电气距离。

## Qwen LoRA 训练

默认配置为 Qwen3.5-9B、rank 32、所有线性层 LoRA、基座严格冻结：

```powershell
& $python scripts\train.py --config configs\default.json --backbone qwen
& $python scripts\evaluate.py `
  --config configs\default.json `
  --checkpoint outputs\default\adapter_checkpoint.pt `
  --backbone qwen
```

只检查本地权重、冻结状态、块图前向和 LoRA 梯度：

```powershell
& $python scripts\qwen_smoke.py --config configs\default.json --backward
```

带结构残差正则的配置：

```powershell
& $python scripts\train.py --config configs\improved.json --backbone qwen
```

轻量随机骨干只用于快速回归测试，不代表 Qwen 最终精度：

```powershell
& $python scripts\train.py --config configs\default.json `
  --backbone tiny --epochs 30 --output-dir outputs\block_reference
```

## 扩展到大系统

`configs/default.json` 中以下参数控制规模：

- `max_neighbors`：每个传感器最多考虑的电气邻居数；
- `lag_order`：动态因果滞后阶数；
- `group_lasso`：整条传感器块边的稀疏强度；
- `l1_penalty`：块内变量关系稀疏强度；
- `n_jobs`：目标传感器回归的 CPU 并行数。

固定特征数、滞后阶数和最大邻居数时，图估计与存储对传感器数量近似线性增长。

## 核心文件

- `matlab/generate_case33bw_timeseries.m`：AC 潮流与电气距离。
- `src/cm_llm/causal.py`：物理约束块稀疏 VAR。
- `src/cm_llm/model/dgp.py`：稀疏变量关系 DGP。
- `src/cm_llm/model/cmllm.py`：Qwen 三模态融合与 LoRA。
- `src/cm_llm/losses.py`：掩码损失和块结构残差。
- `scripts/train.py`、`scripts/evaluate.py`：训练和评估。
