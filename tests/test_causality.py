"""Full Transformer causal-prefix exercise on CUDA/FP32."""

import pytest
import torch

from verify_runtime import make_model


def check_causal_prefix(model, x: torch.Tensor, k: int):
    """改变后缀并比较前缀 logits；外围已设置 eval/no_grad。"""
    # 1. 独立复制整个 [2, 64] 输入，保留原输入。
    x_changed = x.clone()

    # 2. 只改后缀 [:, k:]。加1再取模，保证每个 ID 都改变且仍合法。
    x_changed[:, k:] = (x_changed[:, k:] + 1) % model.vocab_size

    # 3. 同一个模型做两次前向，输出均为 [2, 64, 512]；不更新参数。
    logits_original = model(x)
    logits_changed = model(x_changed)

    # 4. 比较前 k 个位置的全部词表分数，即两个 [2, 32, 512] Tensor。
    torch.testing.assert_close(
        logits_changed[:, :k, :],
        logits_original[:, :k, :],
        rtol=1e-5,
        atol=1e-6,
    )
    return x_changed, logits_original, logits_changed


def test_transformer_lm_causality_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用，因果性尚未验证；请在可访问 GPU 的环境运行。")

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 只初始化一次完整 Transformer；不创建优化器，也不预先训练。
    model = make_model(device, vocab_size=512, context_length=64, d_model=128, d_ff=384)
    model.eval()
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert value.device == device and value.dtype == torch.float32, name

    x = torch.randint(0, 512, (2, 64), device=device, dtype=torch.long)
    k = 32
    original_input = x.clone()
    parameter_snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        x_changed, logits_original, logits_changed = check_causal_prefix(model, x, k)

        # 防止输入别名、未实际修改后缀、越界 ID 等导致无效检查。
        torch.testing.assert_close(x, original_input, rtol=0, atol=0)
        assert x_changed.shape == (2, 64)
        assert x_changed.device == device and x_changed.dtype == torch.long
        assert ((x_changed >= 0) & (x_changed < 512)).all(), "token ID 越界"
        torch.testing.assert_close(x_changed[:, :k], original_input[:, :k], rtol=0, atol=0)
        assert (x_changed[:, k:] != original_input[:, k:]).any(dim=1).all(), "每个样本的后缀必须确实改变"

        for logits in (logits_original, logits_changed):
            assert logits.shape == (2, 64, 512)
            assert logits.device == device and logits.dtype == torch.float32
            assert not logits.requires_grad
            assert torch.isfinite(logits).all(), "logits 出现非有限值"
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter, parameter_snapshot[name], rtol=0, atol=0, msg=f"测试期间参数发生变化：{name}"
            )
