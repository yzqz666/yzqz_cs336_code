# CS336 第一周学习与实验报告

计划周期：2026-10-06 至 2026-10-11。实际记录日期：2026-10-06，时区 Asia/Shanghai。
各历史实验的运行日期未记录，不按计划日期推算完成时间。

## 1. 本周摘要与完成状态

本周围绕项目现有的自定义 Transformer，为 AI Infra 实习准备完成环境恢复、组件测试基线、
CUDA 单步更新、固定批次过拟合和完整模型因果性检查。历史结果表明小规模 CUDA 训练闭环
可以运行，同一合成批次经过200次更新，loss从6.478386降至0.006654。
Day 1—4核心检查及Day 5过拟合核心实验已完成，目前处于收尾阶段。
这些Day标签表示任务阶段，不表示已经实际学习了六天。

| 项目或阶段 | 状态 | 支撑证据及边界 |
| --- | --- | --- |
| Day 1—4：环境恢复、组件基线、单步更新、因果性 | 已完成 | 用户历史运行记录；本次核对当前实现和环境，没有重跑实验 |
| Day 5：固定批次过拟合核心实验 | 已完成 | 用户历史200步结果；当前 `overfit_updates` 与配置已核对 |
| 第一周报告、环境快照、稀疏loss记录 | 已完成 | 本报告及附件由本次文档任务生成 |
| checkpoint详细运行日志 | 待留档 | 用户已确认相关流程；有此前Agent通过汇报及本地文件，原始stdout待补 |
| 完整200步loss历史 | 未留档 | 本地没有找到完整CSV；现有12个历史观测点已留档，不要求为此重训 |
| 新增测试后的最新全量统计 | 尚未验证 | 本次未跑整套测试，不能把历史基线与独立测试结果相加 |
| 3090服务器执行、真实文本训练与泛化、性能优化 | 尚未验证 | 本机合成数据实验不能支持这些结论 |

### 证据约定

| 标记 | 证据来源 | 本报告中的用途 |
| --- | --- | --- |
| R：本次只读检查 | [read_only_checks.json](assets/week1/read_only_checks.json) | 实际环境、Git与文件摘要、已有checkpoint的静态读取结果；没有训练或测试 |
| L：本地既有记录 | [log.txt](../log.txt)、BPE metadata、已有磁盘checkpoint文件 | 证明对应日志或文件存在；不据文件存在推断全部恢复流程通过 |
| H：用户历史运行结果 | [用户历史摘录](assets/week1/user_history_results.md) | 组件基线、单步、过拟合、因果性结果；明确为历史运行，非本次重跑 |
| C：当前代码 | 下文链接的实现、CLI和测试 | 说明配置、检查方法和可运行入口；代码存在不等于运行通过 |

此前会话中的Agent曾汇报checkpoint检查通过，作为额外的会话记录保留；本次未找到其本地
原始运行日志，不把该汇报改写为本次实测结果。本地缺日志也不表示用户没有做过实验。

## 2. 实验环境与代码版本

以下环境来自本次只读查询R，时间为2026-10-06 18:47:53 +08:00。
查询使用当前项目环境 `uv run --no-sync --offline python`，另执行 `uv --version` 与
`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`；未同步依赖。

| 项目 | 当前实测值 |
| --- | --- |
| 运行环境 | Linux / WSL 2，x86_64；内核 `6.18.33.2-microsoft-standard-WSL2` |
| Python解释器 | `/home/yzqz/lessons/cs336/assignment1-basics/.venv/bin/python3` |
| Python版本 | 3.12.3；[.python-version](../.python-version)指定3.12 |
| uv | 0.11.7 |
| PyTorch | `2.7.1+cu128` |
| PyTorch CUDA构建版本 | 12.8；这不是系统CUDA Toolkit版本的测量 |
| CUDA可见性 | `torch.cuda.is_available() = True` |
| GPU | NVIDIA GeForce RTX 5070 Ti；用户历史标称16GB，本次 `nvidia-smi` 报告16303MiB |
| GPU计算能力 | `(12, 0)`，即 `sm_120` |
| NVIDIA驱动 | 610.88 |
| wheel架构列表 | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120` |
| 其他已安装版本 | NumPy 2.3.2、pytest 8.4.1、Triton 3.3.1、tiktoken 0.11.0 |

[pyproject.toml](../pyproject.toml)要求Python >=3.11；当前项目选择3.12。
Linux/Windows的torch源显式指向cu128索引，声明 `torch==2.7.1`；
[uv.lock](../uv.lock)对应平台锁定为 `torch==2.7.1+cu128`。
当前安装版本与这两处配置一致。Intel Mac使用另一条依赖分支，不是本次环境。
[README](../README.md)已有环境和运行说明；其中的服务器安装及传输说明不作为服务器实测证据。

当前Git分支为 `main`，HEAD为 `e1646605865851aa9dbd995d4ff432d0f6704689`。
报告生成前工作区已有如下修改，均保留：

```text
修改：README.md、pyproject.toml、uv.lock
修改：transformer/checkpointing.py、transformer/data_loading.py、transformer/transformer_lm.py
删除：带U+3000前导字符的「　log.txt」；普通log.txt仍存在
未跟踪：.vscode/、encode_dataset.py、tokenizer_experiments.py、verify_runtime.py
未跟踪：tests/test_causality.py、tests/test_runtime.py
```

以上状态来自报告创建前的 `git status --short`，原始输出及44个源文件/环境配置文件的SHA-256
见R附件。报告创建后会新增 `docs/` 文件。**仅检出HEAD不足以还原当前实验代码**；复现需要
保留这些修改及未跟踪源文件。摘要用于识别当前文件，不替代源代码归档，也不证明历史运行时
每个文件的内容均与当前相同。没有自动commit、push或重置工作区。

wheel包含 `sm_86` 只能说明安装包提供该架构目标；本次没有接触3090服务器，不能宣称其运行通过。

## 3. 关键实现与实验方法

### 3.1 训练更新调用关系

```text
make_cuda_experiment：固定seed=42，创建模型并移到CUDA/FP32，再创建AdamW
随机long tokens [2,65]
  → x=tokens[:,:-1]、y=tokens[:,1:]，均为[2,64]
  → train_step：清梯度
  → Transformer(x)：embedding → 2个TransformerBlock → RMSNorm → 词表投影
  → logits [2,64,512]
  → cross_entropy(logits,y)：对batch与序列位置取平均，得到标量loss
  → loss.backward()：使用PyTorch autograd
  → 自定义AdamW.step()：一次参数更新
```

每次更新包含2×64个位置的下一token预测任务，平均为一个loss后执行一次更新。
`d_model=128`是内部表示宽度，`vocab_size=512`决定输出每个位置有512个词表分数。

| 实际文件及接口 | 用途与检查入口 |
| --- | --- |
| [verify_runtime.py](../verify_runtime.py)：`make_model`、`make_optimizer`、`make_cuda_experiment` | 创建固定实验配置；核心练习为 `train_step`，不是另一个 `training_step` |
| [transformer_lm.py](../transformer/transformer_lm.py)：`Transformer(...).forward(x)` | 输出 `[batch, sequence, vocab_size]` logits |
| [transformer_block.py](../transformer/transformer_block.py)：`TransformerBlock` | RMSNorm、因果自注意力、RoPE、SwiGLU及残差连接 |
| [cross_entropy.py](../transformer/cross_entropy.py)：`cross_entropy(inputs, targets)` | 平移logits后计算稳定的交叉熵；直接接收三维logits和二维目标 |
| [AdamW.py](../transformer/AdamW.py)：`AdamW(params, lr, betas, eps, weight_decay)` | 每个参数维护 `m`、`v`、`step`；实现参数更新 |
| [data_loading.py](../transformer/data_loading.py)：`data_loading(array, batch_size, context_length, device)` | 从token数组采样，构造错位输入/目标；uint16存储转成int64输入 |
| [tests/test_model.py](../tests/test_model.py)、[tests/test_nn_utils.py](../tests/test_nn_utils.py)、[tests/test_optimizer.py](../tests/test_optimizer.py) | 模型组件、交叉熵与AdamW等测试；经 [adapters.py](../tests/adapters.py)接入本地实现 |
| [tests/test_runtime.py](../tests/test_runtime.py)：`test_get_batch_from_uint16_memmap` | 检查高token ID从uint16 memmap读取后保持long类型和错位关系；本次仅核对代码 |

### 3.2 单步与固定批次实验配置

下表来自当前代码C，H3/H4已提供的配置字段与之相符。RoPE、betas、eps及TF32设置等
补充参数取自当前实现；历史记录没有逐项提供这些参数，不据当前代码补造历史配置文件。

| 参数 | 值 |
| --- | --- |
| seed、设备、精度 | 42、`cuda:0`、FP32；CUDA matmul与cuDNN的TF32均关闭 |
| vocab_size、batch_size、context_length | 512、2、64 |
| num_layers、d_model、num_heads、d_ff | 2、128、4、384 |
| RoPE max_seq_len、theta | 64、10000.0 |
| AdamW lr、weight_decay | 1e-3、0 |
| AdamW betas、eps | (0.9, 0.999)、1e-8 |
| 输入与目标 | 合成随机long tokens；生成一次 `[2,65]` 后切出固定x/y |

`verify_single_step`使用前向hook检查输出形状/设备/精度，使用step前hook检查梯度缺失、
梯度有限性及设备/精度，并保证只调用一次 `optimizer.step()`。它检查标量loss、更新后参数
有限性，并用独立参数快照确认至少一个参数张量变化；缺失梯度会列出参数名。
不要求每个参数元素都变化，也不把单步loss下降当成断言。

`verify_overfit`在循环外只创建一次模型、优化器、x/y。`overfit_updates`连续调用用户完成的
`train_step`，将loss转换为Python float记录，每20步打印；没有复用单步的“一次更新”hook。
初始与最终评估使用 `evaluate_fixed_batch` 的eval/no_grad，检查logits和loss有限；循环检查每个
记录loss有限，完成后检查各可训练参数的AdamW步数与指定更新次数一致。
本实验验证同一批数据上的可拟合性，不测新数据的泛化。

### 3.3 因果性方法

[tests/test_causality.py](../tests/test_causality.py)中的
`test_transformer_lm_causality_cuda`只初始化一次上述完整模型，不需要先训练。
固定seed=42、CUDA/FP32，eval/no_grad；原输入为long `[2,64]`，`k=32`。
`check_causal_prefix`克隆输入，把后缀 `[:,32:]` 的每个ID改为 `(ID+1)%512`，保证改变且合法。
用相同参数做两次前向，比较 `[:,:32,:]` 的全部logits，容差 `rtol=1e-5, atol=1e-6`。
外围还检查前缀和原输入未改、后缀确实改变、输出有限、参数未更新。

这里使用零基下标：输出位置0—31分别预测后续token，包括位置31对第33个token的预测。
即使后缀中的目标token改变，这些位置的预测分数仍不应依赖它。
实现中的约束位于 [multihead_self_attention.py](../transformer/multihead_self_attention.py)
的下三角mask，以及 [scaled_dot_product_attention.py](../transformer/scaled_dot_product_attention.py)
在softmax前将未来位置分数置为负无穷。
既有 `test_transformer_lm_truncated_input` 是截断输入的参考结果检查，不能直接替代这项后缀扰动实验。

### 3.4 本人参与

依据用户历史说明，本人完成 `train_step` 的训练更新逻辑及 `overfit_updates` 的连续更新、
loss记录循环；Agent协助外围骨架、参数接线与部分检查。其他实现的作者归属未明确记录。
这些实践提供了训练闭环与检查方法的学习证据，不据此宣称整个项目由本人独立从零完成，
也不据测试通过宣称已经熟练掌握所有原理。

## 4. 实验结果与问题记录

### 4.1 环境问题、组件基线、单步与因果性

| 项目 | 历史命令或观测 | 结果与证据来源 |
| --- | --- | --- |
| 5070 Ti兼容问题 | 原环境 `2.6.0+cu124`；CUDA可见但不支持 `sm_120` | H1：实际计算报错 `CUDA error: no kernel image is available for execution on the device` |
| 环境处理结果 | 当前依赖声明、锁文件、安装版本均核对 | R：现装 `2.7.1+cu128`、CUDA构建12.8、架构列表含 `sm_120`；H3/H4训练成功支撑基础计算可用，本次没有重新计算 |
| 当时组件测试基线 | `uv run pytest -q --tb=short` | H2：48 PASSED、1 XPASS，无FAILED/ERROR；不是当前最新全量统计 |
| CUDA单步更新 | `uv run --locked verify_runtime.py --single-step` | H3：loss=6.478386，21个参数张量变化，检查通过；loss在更新前计算 |
| 固定批次过拟合 | `uv run --locked verify_runtime.py --overfit --steps 200` | H4：完成200次更新，初始6.478386，最终重新评估0.006654 |
| 完整Transformer因果性 | `uv run --locked pytest tests/test_causality.py::test_transformer_lm_causality_cuda -q` | H5：`1 passed in 1.99s`；只支持这组条件和数值容差内的前缀不变结论 |

本周已实际遇到的问题是安装包的GPU架构兼容性。`CUDA available=True`只说明CUDA可见，
不能单独证明所安装wheel能执行该GPU的计算。调整后的基础训练能力有历史单步与过拟合结果支持。

XPASS项目为 [tests/test_tokenizer.py](../tests/test_tokenizer.py)中的 `test_encode_memory_usage`。
它标注了预期失败的内存限制说明；其中1MB不是本次实测内存数字。保留原测试预期，
不通过改断言或标记来改变结果。独立因果性测试与旧组件基线不直接相加。

### 4.2 固定批次loss：用户历史日志摘录

本地没有找到这次过拟合的完整stdout、CSV或图片。以下数据全部来自用户提供的H4，
已保存为 [historical_overfit_sparse.csv](assets/week1/historical_overfit_sparse.csv)。
**横轴定义为已完成的参数更新次数**，第n次更新前对应n-1次，最终评估对应200次。

| 历史记录标签 | 已完成更新次数 | loss |
| --- | ---: | ---: |
| 初始评估 | 0 | 6.478386 |
| 第20次更新前 | 19 | 0.517197 |
| 第40次更新前 | 39 | 0.065005 |
| 第60次更新前 | 59 | 0.030461 |
| 第80次更新前 | 79 | 0.020883 |
| 第100次更新前 | 99 | 0.016074 |
| 第120次更新前 | 119 | 0.012949 |
| 第140次更新前 | 139 | 0.010709 |
| 第160次更新前 | 159 | 0.009028 |
| 第180次更新前 | 179 | 0.007728 |
| 第200次更新前 | 199 | 0.006699 |
| 完成200次更新后重新评估 | 200 | 0.006654 |

由历史观测计算：`(6.478386-0.006654)/6.478386×100% = 99.89728923222543%`，
约为 **99.8973%的loss下降比例**，不是准确率。
第200次更新前的0.006699与更新后评估的0.006654是两个不同观测。

图片未生成：本次只读检查发现当前项目环境没有Matplotlib、Plotly、Bokeh、Seaborn，
也没有gnuplot。按任务约定保留CSV与结果表，没有为绘图安装或升级依赖。
仅有12个真实采样点，没有插值补齐200步；本次没有新训练曲线，历史与新运行数据未混合。

### 4.3 checkpoint：代码范围、文件证据与运行证据分开记录

[transformer/checkpointing.py](../transformer/checkpointing.py)已实现
`save_checkpoint(model, optimizer, iteration, out)`和 `load_checkpoint(src, model, optimizer)`。
保存三项：模型state_dict、优化器state_dict和全局iteration。
优化器包含参数组超参数及每个参数的 `m`（梯度移动平均）、`v`（梯度平方移动平均）、
`step`（偏差修正所需的更新计数）。外层iteration记录全局训练进度。
仅恢复权重会丢失优化器历史，后续更新方向或幅度可能变化，不能保证与不中断训练的轨迹一致。

当前运行入口已经提供下面的检查，属于C：

| 范围 | 实际代码或文件 | 本次确认程度 |
| --- | --- | --- |
| 保存/加载核心接口 | `save_checkpoint`、`load_checkpoint`；加载映射到新模型所在设备 | 已阅读实现；没有要求重写核心函数 |
| 磁盘文件与同进程序列化 | [tests/test_serialization.py](../tests/test_serialization.py)：`test_checkpointing` | 小型 `_TestNet` 的循环配置为10步，写临时磁盘文件，比较模型与完整优化器状态及iteration；本次未运行该测试 |
| BytesIO、恢复前后logits | `verify_runtime.py::verify` | 代码检查iteration和logits完全一致（rtol=atol=0）；本地原始运行日志未找到，本次未执行 |
| 恢复后再更新与不中断更新对照 | `verify`调用原 `training_step`，对两条路径各更新一次，再逐项比较模型state_dict | 检查存在；不是只判断能加载文件；本次未执行 |
| GPU→CPU恢复后更新 | `verify`的CUDA分支，CPU上重建模型/优化器、加载BytesIO并训练一步 | 检查存在；未要求CPU/GPU输出逐元素一致；本次未执行跨设备训练 |
| 磁盘保存及新Python进程恢复 | `write_disk_checkpoint`写文件；`verify`用 `subprocess.run` 启动 `--restore-checkpoint` | 入口存在；本地文件静态检查已完成，新进程恢复本次未执行 |
| 新进程步数、logits及续训一步 | `restore_disk_checkpoint`检查PID不同、iteration、各AdamW step、参考logits完全一致，然后更新一步并检查参数变化 | 检查存在；本地完整stdout未找到，运行详情待留档 |

checkpoint原验证使用另一组已有小模型配置：seed=42、2层、4头、词表128、序列16、
batch=2、d_model=32、d_ff=64、RoPE theta=10000.0、max_seq_len=16；
AdamW lr=1e-3、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01。
数据为 `np.arange(1024)%128` 的uint16数组，经 `data_loading`构造同一批输入/目标。
原 `training_step`还使用max_l2_norm=1.0的梯度裁剪。它与512词表的 `train_step`练习配置分别记录，
不能将其checkpoint当作200步过拟合模型的保存产物。

本次R于2026-10-06 18:48:44 +08:00，只读检查已有目录 `/tmp/cs336-week1-checkpoint`：

| 文件或字段 | 实际读到的结果 |
| --- | --- |
| `checkpoint.pt` | 文件存在，366886字节；使用 `torch.load(..., map_location="cpu", weights_only=True)` 成功读取 |
| checkpoint顶层字段 | `model_state_dict`、`optimizer_state_dict`、`iteration` |
| 保存iteration与优化器状态 | iteration=3；21组参数状态均包含m/v/step，step均为3；m/v张量均为有限值 |
| 保存优化器参数组 | lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01 |
| `reference.pt` | 文件存在，19315字节；保存的设备标签为 `cuda:0`，参考iteration=3 |
| 参考输入与输出 | inputs/targets均为int64 `[2,16]`；logits为FP32 `[2,16,128]`且有限；模型kwargs与上述小模型一致 |

这证明文件已落盘且可读，保存了相应状态。**本次没有重建模型、重新比较logits、恢复后更新，
也没有执行新进程恢复。**将文件映射到CPU做静态检查，不等于GPU checkpoint在CPU恢复训练通过。
PID字段的存在也不等于新进程实验本次通过。
用户已确认相关流程、此前Agent已汇报通过，这些会话证据保留；各运行范围的本地原始日志待补，
不以“已核对”推断全部通过。

`reference.pt`是该实验另存的配置、固定批次和logits参考，不是核心checkpoint的第四个字段。
核心checkpoint不保存随机数状态或数据采样进度；模型结构需按相同配置重建。
[RoPE.py](../transformer/RoPE.py)中的sin/cos buffer为 `persistent=False`，由初始化配置重新构造。
本检查范围不能推广到任意数据采样或跨环境的完整训练轨迹复现。
文件保留在原临时目录，报告只收录小型检查摘要，没有复制模型权重；临时目录不能替代长期留档。

### 4.4 其他已有本地材料

[log.txt](../log.txt)是BPE训练记录，不是单步/过拟合/checkpoint日志。
[train_bpe的metadata](../artifacts/tinystories_bpe/train_bpe/metadata.json)和
[train_bpe_v2的metadata](../artifacts/tinystories_bpe/train_bpe_v2/metadata.json)均记录词表10000、
合并9743次；分别记录耗时约547.56s和162.00s，与本地日志一致，来源为L。
这些记录的具体运行日期未记录，不将BPE耗时作为Transformer训练性能。

本次文件检查还确认已有TinyStories训练/验证原文以及 `data/tinystories_train.bin`、
`data/tinystories_valid.bin`，大小见R附件。[encode_dataset.py](../encode_dataset.py)已有流式编码
入口，默认读取验证原文和 `train_bpe_v2` 词表/merges，输出uint16 token文件。
文件存在证明已有产物，不能单独证明编码全过程、训练/验证分割和真实文本训练已经验证完成。
本次没有读取大型数据内容，也没有运行编码。

## 5. 复现步骤

以下命令从项目根目录执行。除上文列出的只读查询外，**这些复现实验命令本次均未执行**。
历史命令原样保留在H附件及第4节；当前建议使用 `uv run --no-sync --offline`复用已安装环境，
避免文档收尾时触发同步。`--no-sync`不会自动校验环境与锁文件一致，因此第2节另行核对了安装版本。

### 5.1 前置条件及环境核对

已有项目 `.venv`、本地源文件及测试fixtures；CUDA实验要求运行进程能访问GPU。
迁移到新机器时需要依据 [README](../README.md)单独准备锁定环境和驱动，不能只复制本地 `.venv`。
本次没有安装、升级或访问外部资料。可用以下只读命令重新查看环境：

```bash
export UV_CACHE_DIR=/tmp/cs336-uv-cache
TZ=Asia/Shanghai date -Iseconds
uv run --no-sync --offline python -c 'import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git branch --show-current
git rev-parse HEAD
git status --short
```

上述UV缓存目录是本机约定，其他机器可使用其可写缓存目录。
单步、过拟合、因果性和原runtime检查都使用合成数据，不需要下载TinyStories。
组件全量测试的一部分调用 `tiktoken.get_encoding("gpt2")`，需要参考tokenizer缓存；
本机 `/tmp/cs336-tiktoken-cache`目录存在，但本次未重测其全部内容。
`uv --offline`仅控制uv的联网行为，不会禁止Python库自行下载缺失参考文件。
迁移后需要另行准备缓存或满足测试的下载条件。

### 5.2 重跑核心实验

```bash
# 单步CUDA更新
uv run --no-sync --offline verify_runtime.py --single-step

# 固定批次200次更新
uv run --no-sync --offline verify_runtime.py --overfit --steps 200

# 完整Transformer因果性检查
uv run --no-sync --offline pytest tests/test_causality.py::test_transformer_lm_causality_cuda -q
```

`--single-step`和 `--overfit`互斥，均在函数内部固定CUDA设备；不需要额外传 `--device cuda`。
`--steps`控制overfit更新次数，单步模式始终只更新一次。
因果性测试在CUDA不可用时会显示skip，skip不代表因果性通过。
新运行的数值或耗时若与H不同，应单独记录，不覆盖历史数据。

### 5.3 checkpoint复现与留档

```bash
# 既有磁盘序列化单元测试，同一Python进程
uv run --no-sync --offline pytest tests/test_serialization.py::test_checkpointing -q

# 原BytesIO训练/恢复/继续更新流程；CUDA分支也检查CPU恢复后更新
uv run --no-sync --offline verify_runtime.py --device cuda --steps 3

# 额外写磁盘文件，并自动启动新的Python进程做恢复检查
uv run --no-sync --offline verify_runtime.py --device cuda --steps 3 --checkpoint-dir /tmp/cs336-week1-rerun

# 上一条成功产生checkpoint.pt和reference.pt后，可单独重做新进程恢复检查
uv run --no-sync --offline verify_runtime.py --restore-checkpoint /tmp/cs336-week1-rerun
```

这些参数来自当前CLI，没有 `--resume`选项。`--checkpoint-dir`只适用于原runtime模式，
不能与 `--single-step`、`--overfit`、`--restore-checkpoint`合用。
`--restore-checkpoint`依赖两份文件，使用 `reference.pt`记录的设备和配置。
建议使用新的输出目录保留旧文件；核心文件保存于第3次更新完成后，后续检查会再更新一步。

需要补充本地stdout时，可独立保存重跑结果，例如：

```bash
mkdir -p docs/assets/week1/reruns
set -o pipefail
uv run --no-sync --offline verify_runtime.py --device cuda --steps 3 --checkpoint-dir /tmp/cs336-week1-rerun \
  2>&1 | tee docs/assets/week1/reruns/checkpoint_stdout.txt
```

这一操作会重新执行原小模型验证，报告任务没有自动执行它。
`reruns`与历史CSV分别存放，避免把新运行结果混入历史记录。

### 5.4 可选的最新全量统计

```bash
TIKTOKEN_CACHE_DIR=/tmp/cs336-tiktoken-cache uv run --no-sync --offline pytest -q --tb=short
```

本次未执行；运行后应记录实际日期、命令、passed/xpass/skipped/failed统计，
不能继续使用历史48 PASSED、1 XPASS作为新增测试后的全量结果。

## 6. 结论、局限与下一步

当前证据支持：现有自定义Transformer、交叉熵、AdamW能够完成小规模CUDA训练闭环；
固定合成数据上的loss明显下降；在seed=42、k=32及给定容差内，后缀变化未影响前缀logits。
这些检查分别覆盖训练路径、固定数据可拟合性与未来信息隔离，作用互补。

结论边界包括：本次是文档与只读核对，没有重新训练或重新跑测试；历史组件统计不是最新全量；
固定批次拟合不代表真实语言模型训练完成、生成质量或泛化能力；因果性测试不是对全部输入的数学证明。
checkpoint文件状态已可检查，但详细运行日志仍待留档；没有验证3090服务器、任意采样状态恢复、
吞吐量优化或长期训练稳定性。

下一阶段聚焦现有文本流程：

1. 核对 [encode_dataset.py](../encode_dataset.py)的真实文本编码产物、token范围和uint16读取方式，
   记录输入/词表/merges及输出统计。现有BPE词表为10000，后续模型词表配置需与编码ID范围匹配。
2. 核对训练/验证原文与编码文件的划分及来源，避免重复使用训练批次充当验证集。
3. 在这些前置检查后开展短训练，分别记录训练loss和独立验证loss，完整保存命令、配置及日志。
4. 使用训练后的模型做短文本生成，记录提示、生成设置和实际输出，再评估下一阶段需要的改进。

本次只生成报告及必要的小型文档附件；没有修改模型、训练逻辑、测试断言、依赖或锁文件，
没有新增真实训练系统、调度器、混合精度、分布式、vLLM或CUDA内核任务。

交付检查已完成：35处Markdown相对链接均有效；CSV的12个loss值及已完成更新次数与H4逐项一致；
`uv run --no-sync --offline verify_runtime.py --help`已实际执行，退出码0，确认文中使用的CLI选项存在；
44个源文件/环境配置文件与报告生成前的SHA-256一致，`git diff --check`通过。
这些是文档和只读检查的结果，不计入模型测试通过数量。
