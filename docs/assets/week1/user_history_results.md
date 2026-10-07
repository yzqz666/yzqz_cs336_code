# 用户历史运行结果摘录

整理日期：2026-10-06，Asia/Shanghai。来源：本次报告任务中用户提供的真实历史运行记录。
各实验的实际运行日期未记录。本文不是原始 stdout 文件，也不是本次重新执行产生的日志。

## H1：环境兼容问题

本机 RTX 5070 Ti 16GB，项目在 WSL 2 中运行。原环境 PyTorch 2.6.0+cu124；
`CUDA available=True`，显卡架构 `sm_120`，安装包提示不支持该架构，实际计算失败：

```text
CUDA error: no kernel image is available for execution on the device
```

后来环境调整后，GPU 单步计算和训练实验成功。当前安装版本以
[本次只读检查](read_only_checks.json)为准，不以历史安装建议为准。

## H2：当时的组件测试基线

```bash
uv run pytest -q --tb=short
```

用户粘贴结果：48 项 PASSED、1 项 XPASS，没有 FAILED 或 ERROR。
XPASS 为 `test_encode_memory_usage`。其预期失败说明中的 1MB 不是实测内存。
这不是新增因果性等测试之后的最新全量统计。

## H3：CUDA 单步更新

```bash
uv run --locked verify_runtime.py --single-step
```

配置：seed=42，CUDA/FP32，词表512，batch=2，序列64，2层，d_model=128，
num_heads=4，d_ff=384，learning_rate=1e-3，weight_decay=0。

结果：更新前计算的 loss=6.478386；21 个参数张量发生变化；CUDA 单次训练更新检查通过。
21 是参数张量数量，不是标量参数总数。

## H4：固定批次过拟合

```bash
uv run --locked verify_runtime.py --overfit --steps 200
```

沿用 H3 配置，始终复用同一模型、优化器和固定输入/目标。

```text
初始 loss（0次更新）：6.478386
第20次更新前 loss：0.517197
第40次更新前 loss：0.065005
第60次更新前 loss：0.030461
第80次更新前 loss：0.020883
第100次更新前 loss：0.016074
第120次更新前 loss：0.012949
第140次更新前 loss：0.010709
第160次更新前 loss：0.009028
第180次更新前 loss：0.007728
第200次更新前 loss：0.006699
最终 loss（完成200次更新后重新评估）：0.006654
```

以上仅有12个观测点。第 n 次更新前对应已完成 n-1 次更新。
结构化数据见 [稀疏观测 CSV](historical_overfit_sparse.csv)。没有插值，也没有补造200步历史。

## H5：完整 Transformer 因果性检查

```bash
uv run --locked pytest tests/test_causality.py::test_transformer_lm_causality_cuda -q
```

```text
1 passed in 1.99s
```

条件：CUDA/FP32，固定种子，输入[2,64]、词表512、k=32；克隆输入，仅修改
`[:,32:]`；同一模型、同一参数、eval/no_grad；比较输出`[:,:32,:]`全部logits；
rtol=1e-5，atol=1e-6。

结论范围：在这组条件和容差内，改变输入后缀未影响前缀logits。

## H6：本人参与及 checkpoint 确认范围

用户说明本人亲自补写并提交过 `train_step` 的更新逻辑以及 `overfit_updates` 的
连续更新、loss记录循环。Agent协助实验外围骨架、参数接线及部分检查。
其他实现的作者分工没有明确证据，不作推断。

用户确认已经核对 checkpoint 相关流程，但本次提示没有提供它的完整运行日志。
此前会话中的 Agent 有通过汇报；它不是本次重新执行结果，本地原始 stdout 尚未找到。
这两类确认不能单独作为磁盘、新进程、CPU/GPU跨设备等每个范围均已通过的原始证据。
当前代码覆盖范围和实际文件检查结果见 [主报告](../../week1.md)。
