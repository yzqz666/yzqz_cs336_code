import numpy as np
import torch

from transformer.data_loading import data_loading


def test_get_batch_from_uint16_memmap(tmp_path):
    # Preserve IDs above the signed int16 range when reading the encoded file.
    path = tmp_path / "tokens.bin"
    np.arange(65_000, 65_128, dtype=np.uint16).tofile(path)
    dataset = np.memmap(path, dtype=np.uint16, mode="r")
    inputs, targets = data_loading(dataset, batch_size=8, context_length=16, device="cpu")

    assert inputs.shape == targets.shape == (8, 16)
    assert inputs.dtype == targets.dtype == torch.long
    assert inputs.min() >= 65_000
    assert targets.max() < 65_128
    torch.testing.assert_close(targets, inputs + 1)
