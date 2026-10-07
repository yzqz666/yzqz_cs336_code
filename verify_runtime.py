"""Runtime checks and CUDA exercises via --single-step or --overfit."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from transformer.AdamW import AdamW
from transformer.checkpointing import load_checkpoint, save_checkpoint
from transformer.cross_entropy import cross_entropy
from transformer.data_loading import data_loading
from transformer.gradient_clipping import gradient_clipping
from transformer.transformer_lm import Transformer


def make_model(
    device: torch.device,
    *,
    vocab_size: int = 128,
    context_length: int = 16,
    d_model: int = 32,
    d_ff: int = 64,
) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=2,
        d_model=d_model,
        num_heads=4,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=10_000.0,
    ).to(device=device, dtype=torch.float32)


def make_optimizer(model: Transformer, *, weight_decay: float = 0.01) -> AdamW:
    return AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
    )


def train_step(
    model: Transformer, optimizer: AdamW, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """返回 CUDA / FP32 的零维 loss Tensor；该 loss 在本次参数更新前计算。

    中文伪代码：
    1. 清除优化器管理的参数梯度。
    2. 调用模型得到 x 对应的 logits。
    3. 用已有交叉熵计算 logits 与 y 的平均 loss。
    4. 从 loss 反向传播。
    5. 让已有 AdamW 更新参数一次。
    6. 返回标量 loss Tensor（可以 detach，保持设备和 dtype）。

    调用接口：
    optimizer.zero_grad(set_to_none=True) -> None
    model(x: LongTensor[2, 64]) -> FloatTensor[2, 64, 512]
    cross_entropy(inputs=logits, targets=y) -> FloatTensor[]
    loss.backward() -> None
    optimizer.step() -> None
    """

    optimizer.zero_grad()
    logits = model(x)

    loss = cross_entropy(logits,y)

    loss.backward()
    optimizer.step()
    return loss


def make_cuda_experiment() -> tuple[Transformer, AdamW, torch.Tensor, torch.Tensor]:
    """两个练习复用同一配置，各自只初始化一次模型、优化器和固定批次。"""
    if not torch.cuda.is_available():
        raise RuntimeError("尚未验证：CUDA 不可用，请检查当前运行环境。")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # make_model 先将模型和 RoPE 缓冲区移到 CUDA，再建立优化器。
    model = make_model(device, vocab_size=512, context_length=64, d_model=128, d_ff=384)
    model.train()
    optimizer = make_optimizer(model, weight_decay=0.0)
    tokens = torch.randint(0, 512, (2, 65), device=device, dtype=torch.long)
    return model, optimizer, tokens[:, :-1], tokens[:, 1:]


def verify_single_step() -> None:
    """固定配置的单步练习；仅调用一次 train_step，不执行原验证流程。"""
    model, optimizer, x, y = make_cuda_experiment()
    device = x.device
    parameters = dict(model.named_parameters())
    for name, value in (*parameters.items(), *model.named_buffers()):
        if value.device != device or value.dtype != torch.float32:
            raise RuntimeError(f"参数或缓冲区设备/dtype 错误：{name}: {value.device}, {value.dtype}")

    for name, value in (("x", x), ("y", y)):
        if value.shape != (2, 64) or value.device != device or value.dtype != torch.long:
            raise RuntimeError(f"{name} 的形状、设备或 dtype 不正确")

    # clone 创建独立存储；更新不能同时改变这份快照。
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    output_checked = False
    step_calls = 0

    def check_output(module, args, logits):
        nonlocal output_checked
        if not isinstance(logits, torch.Tensor) or logits.shape != (2, 64, 512):
            raise RuntimeError("logits 必须是形状为 [2, 64, 512] 的 Tensor")
        if logits.device != device or logits.dtype != torch.float32:
            raise RuntimeError(f"logits 设备/dtype 错误：{logits.device}, {logits.dtype}")
        output_checked = True

    def check_gradients():
        missing, invalid = [], []
        for name, parameter in parameters.items():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                missing.append(name)
            elif grad.device != device or grad.dtype != torch.float32 or not torch.isfinite(grad).all():
                invalid.append(name)
        errors = []
        if missing:
            errors.append("缺失梯度的参数：\n" + "\n".join(missing))
        if invalid:
            errors.append("梯度非有限或设备/dtype 错误的参数：\n" + "\n".join(invalid))
        if errors:
            raise RuntimeError("\n".join(errors))

    def check_before_update(optim, args, kwargs):
        nonlocal step_calls
        step_calls += 1
        if step_calls > 1:
            raise RuntimeError("此实验只允许一次 optimizer.step()")
        check_gradients()

    # 在核心函数内部实际前向/更新时检查，避免另做一次前向或更新。
    output_hook = model.register_forward_hook(check_output)
    step_hook = optimizer.register_step_pre_hook(check_before_update)
    print("单步实验：seed=42，CUDA/FP32，词表512，batch=2，序列64。")
    print("初始化和输入检查完成；核心训练更新尚未验证。")
    try:
        loss = train_step(model, optimizer, x, y)
    finally:
        output_hook.remove()
        step_hook.remove()

    if not isinstance(loss, torch.Tensor) or loss.shape != torch.Size([]):
        raise RuntimeError("train_step 必须返回零维 loss Tensor")
    if loss.device != device or loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise RuntimeError("loss 必须是 CUDA/FP32 的有限值")
    check_gradients()
    if not output_checked:
        raise RuntimeError("未检查到模型输出，请在 train_step 中通过 model(x) 调用前向")
    if step_calls != 1:
        raise RuntimeError(f"应更新一次参数，实际 optimizer.step() 次数为 {step_calls}")
    invalid_parameters = [name for name, p in parameters.items() if not torch.isfinite(p).all()]
    if invalid_parameters:
        raise RuntimeError("更新后非有限的参数：\n" + "\n".join(invalid_parameters))
    changed = [name for name, p in parameters.items() if not torch.equal(before[name], p.detach())]
    if not changed:
        raise RuntimeError("更新后没有任何参数发生变化")
    torch.cuda.synchronize(device)
    print(f"PASS：CUDA 单次训练更新已验证，loss={loss.item():.6f}，{len(changed)} 个参数张量发生变化。")


def loss_to_float(loss: torch.Tensor) -> float:
    """检查标量 loss 并转换为 float，避免在历史记录中保留计算图。"""
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise RuntimeError("loss 必须是零维 Tensor")
    if loss.device != torch.device("cuda:0") or loss.dtype != torch.float32:
        raise RuntimeError("loss 必须使用 CUDA/FP32")
    if not torch.isfinite(loss):
        raise RuntimeError("loss 出现非有限值")
    return loss.detach().item()


@torch.no_grad()
def evaluate_fixed_batch(model: Transformer, x: torch.Tensor, y: torch.Tensor) -> float:
    """用当前参数评估固定批次；不计算梯度，也不更新参数。"""
    was_training = model.training
    model.eval()
    try:
        logits = model(x)
        if logits.shape != (2, 64, 512) or logits.device != x.device or logits.dtype != torch.float32:
            raise RuntimeError("评估 logits 的形状、设备或 dtype 不正确")
        if not torch.isfinite(logits).all():
            raise RuntimeError("评估 logits 出现非有限值")
        return loss_to_float(cross_entropy(logits, y))
    finally:
        model.train(was_training)


def overfit_updates(
    model: Transformer, optimizer: AdamW, x: torch.Tensor, y: torch.Tensor, steps: int
) -> list[float]:
    """返回长度为 steps 的更新前 loss 历史。

    中文伪代码：
    - 连续更新 steps 次，始终复用传入的 model、optimizer、x、y。
    - 每次调用已完成的 train_step，而不是 training_step。
    - 用 loss_to_float(loss) 检查有限值、转换为 float，再加入历史记录。
    - 每20步打印一次步数和 loss，明确标注为“本次更新前 loss”。
    - 返回 loss 历史；模型和优化器状态继续保留在传入的对象中。

    第 k 次返回的 loss 在第 k 次更新前计算，模型已完成前 k-1 次更新。
    本函数不负责初始/最终评估，也不重新创建模型、优化器或批次。
    """
    res = []
    for i in range(steps):
        loss = train_step(model,optimizer,x,y)
        res.append(loss_to_float(loss))
        if (i + 1) % 20 == 0:
            print(f"第{i + 1}次更新前 loss: {res[-1]:.6f}")
    return res

def verify_overfit(steps: int) -> None:
    """固定批次过拟合入口：初始化一次，评估初始 loss，再执行练习循环。"""
    model, optimizer, x, y = make_cuda_experiment()
    print(f"固定批次过拟合：seed=42，CUDA/FP32，词表512，batch=2，序列64，更新{steps}次。")
    initial_loss = evaluate_fixed_batch(model, x, y)
    print(f"初始 loss（0次更新）：{initial_loss:.6f}")

    loss_history = overfit_updates(model, optimizer, x, y, steps)
    if not isinstance(loss_history, list) or len(loss_history) != steps:
        raise RuntimeError(f"loss 历史必须是长度为 {steps} 的 list[float]")
    if not all(isinstance(value, float) for value in loss_history) or not np.isfinite(loss_history).all():
        raise RuntimeError("loss 历史必须只包含有限的 float 值，请使用 loss_to_float(loss)")
    # 当前自写 AdamW 为每个参与更新的参数保存 state['step']，无需单步限制 hook。
    wrong_steps = {
        name: optimizer.state.get(parameter, {}).get("step", 0)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and optimizer.state.get(parameter, {}).get("step", 0) != steps
    }
    if wrong_steps:
        raise RuntimeError(f"以下参数的更新次数与 {steps} 不符：{wrong_steps}")

    final_loss = evaluate_fixed_batch(model, x, y)
    print(f"最终 loss（完成{steps}次更新后重新评估）：{final_loss:.6f}")
    print(f"初始到最终 loss 变化：{initial_loss:.6f} -> {final_loss:.6f}")


def training_step(model, optimizer, inputs, targets) -> float:
    optimizer.zero_grad()
    logits = model(inputs)
    if logits.shape != (*inputs.shape, 128):
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    loss = cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    gradient_clipping(list(model.parameters()), max_l2_norm=1.0)
    optimizer.step()
    return loss.item()


def write_disk_checkpoint(directory, checkpoint, model, inputs, targets, iteration) -> None:
    """把现有 BytesIO checkpoint 写入磁盘，另存固定批次和比较参考。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoint.pt").write_bytes(checkpoint.getvalue())
    with torch.no_grad():
        logits = model(inputs).cpu()
    torch.save(
        {
            "model_kwargs": {
                "vocab_size": model.vocab_size,
                "context_length": model.context_length,
                "d_model": model.norm.d_model,
                "d_ff": model.transformer_blocks[0].ffn.d_ff,
            },
            "device": str(inputs.device),
            "inputs": inputs.cpu(),
            "targets": targets.cpu(),
            "logits": logits,
            "iteration": iteration,
            "training": model.training,
            "parent_pid": os.getpid(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        directory / "reference.pt",
    )
    print(f"Disk checkpoint and fixed-batch reference saved: {directory}", flush=True)


def restore_disk_checkpoint(directory: Path) -> None:
    """新进程复用现有恢复函数，检查步数、logits 和恢复后的单次更新。"""
    reference = torch.load(directory / "reference.pt", map_location="cpu", weights_only=True)
    if os.getpid() == reference["parent_pid"]:
        raise RuntimeError("This check must run in a new Python process")
    device = torch.device(reference["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; disk checkpoint restore has not been verified")
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(reference["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = reference["cuda_tf32"]
    torch.backends.cudnn.allow_tf32 = reference["cudnn_tf32"]

    model = make_model(device, **reference["model_kwargs"])
    model.train(reference["training"])
    optimizer = make_optimizer(model)
    inputs = reference["inputs"].to(device)
    targets = reference["targets"].to(device)
    iteration = load_checkpoint(directory / "checkpoint.pt", model, optimizer)
    assert iteration == reference["iteration"], "Checkpoint iteration was not restored"
    assert optimizer.state, "AdamW history was not restored"
    for state in optimizer.state.values():
        assert state["step"] == iteration, "AdamW step was not restored"
    print(f"New Python process: PID={os.getpid()}, saved by PID={reference['parent_pid']}")
    print(f"PASS: restored iteration={iteration}, AdamW step={iteration}")

    with torch.no_grad():
        torch.testing.assert_close(model(inputs), reference["logits"].to(device), rtol=0, atol=0)
    print("PASS: restored logits match disk reference exactly (rtol=0, atol=0)")

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    # 沿用原 verify() 的小模型配置和 training_step，保持其训练逻辑不变。
    loss = training_step(model, optimizer, inputs, targets)
    iteration += 1
    for state in optimizer.state.values():
        assert state["step"] == iteration, "AdamW did not advance by one step"
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters()), "No parameter changed"
    print(f"PASS: resumed one update; iteration={iteration}, AdamW step={iteration}, loss={loss:.6f}")


def verify(device: torch.device, steps: int, checkpoint_dir: Path | None = None) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"PyTorch: {torch.__version__}; wheel CUDA: {torch.version.cuda}")
    print(f"Device: {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch wheel.")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
        print(f"Wheel architectures: {torch.cuda.get_arch_list()}")

    model = make_model(device)
    optimizer = make_optimizer(model)
    # Use the same uint16 storage format as encode_dataset.py.
    dataset = (np.arange(1024) % model.vocab_size).astype(np.uint16)
    inputs, targets = data_loading(dataset, batch_size=2, context_length=16, device=device)
    before = model.embedding.embedding_matrix.detach().clone()
    for step in range(steps):
        loss = training_step(model, optimizer, inputs, targets)
        print(f"Step {step + 1}/{steps}: loss={loss:.6f}")
    if torch.equal(before, model.embedding.embedding_matrix):
        raise RuntimeError("Optimizer did not update the model weights")

    checkpoint = io.BytesIO()
    save_checkpoint(model, optimizer, steps, checkpoint)
    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir.resolve()
        write_disk_checkpoint(checkpoint_dir, checkpoint, model, inputs, targets, steps)
    checkpoint.seek(0)
    restored_model = make_model(device)
    restored_optimizer = make_optimizer(restored_model)
    if load_checkpoint(checkpoint, restored_model, restored_optimizer) != steps:
        raise RuntimeError("Checkpoint iteration was not restored")
    with torch.no_grad():
        torch.testing.assert_close(restored_model(inputs), model(inputs), rtol=0, atol=0)

    # A resumed step must match uninterrupted training, including AdamW state.
    training_step(model, optimizer, inputs, targets)
    training_step(restored_model, restored_optimizer, inputs, targets)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name], value, rtol=0, atol=0)

    # A GPU checkpoint must also load on CPU for local debugging.
    if device.type == "cuda":
        checkpoint.seek(0)
        cpu_model = make_model(torch.device("cpu"))
        cpu_optimizer = make_optimizer(cpu_model)
        if load_checkpoint(checkpoint, cpu_model, cpu_optimizer) != steps:
            raise RuntimeError("CPU checkpoint iteration was not restored")
        cpu_loss = training_step(cpu_model, cpu_optimizer, inputs.cpu(), targets.cpu())
        print(f"GPU checkpoint resumed on CPU: loss={cpu_loss:.6f}")
        torch.cuda.synchronize(device)
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB")
    print("PASS: forward, backward, optimizer update, checkpoint restore and resumed training")
    if checkpoint_dir is not None:
        print("Starting a new Python process for disk checkpoint restore", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--restore-checkpoint", str(checkpoint_dir)],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--single-step", action="store_true", help="固定 CUDA/FP32 的单步验证")
    modes.add_argument("--overfit", action="store_true", help="固定批次过拟合练习")
    modes.add_argument("--restore-checkpoint", type=Path, help="在新进程恢复目录中的 checkpoint.pt，并验证续训一步")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--steps", type=int, default=3, help="更新次数，用于 --overfit 或原运行验证模式")
    parser.add_argument("--checkpoint-dir", type=Path, help="原运行验证模式额外保存磁盘 checkpoint，并自动启动新进程恢复")
    args = parser.parse_args()
    if args.checkpoint_dir is not None and (args.single_step or args.overfit or args.restore_checkpoint is not None):
        parser.error("--checkpoint-dir is only supported by the original runtime verification mode")
    if args.restore_checkpoint is not None:
        restore_disk_checkpoint(args.restore_checkpoint.resolve())
        return
    if args.single_step:
        try:
            verify_single_step()
        except NotImplementedError as error:
            parser.exit(1, f"{error}\n")
        return
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.overfit:
        try:
            verify_overfit(args.steps)
        except NotImplementedError as error:
            parser.exit(1, f"{error}\n")
        return
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    verify(torch.device(device), args.steps, args.checkpoint_dir)


if __name__ == "__main__":
    main()
