import torch
import numpy as np

def data_loading(array, batch_size, context_length, device):
    starts = np.random.randint(0, len(array) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)
    # Encoded datasets use uint16 on disk; embeddings and loss need int64 IDs.
    train = torch.tensor(array[indices].astype(np.int64), device=device)
    val = torch.tensor(array[indices + 1].astype(np.int64), device=device)
    return train, val
