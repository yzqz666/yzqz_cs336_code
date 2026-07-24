import torch
import numpy as np

def data_loading(array,batch_size,context_length,device):
    train = []
    val = []
    for i in range(batch_size):
        start = np.random.randint(0,len(array) - context_length)
        x = np.stack(array[start:start + context_length])
        y = np.stack(array[start + 1:start + context_length+1])
        train.append(x)
        val.append(y)
    train = torch.tensor(train,device = device)
    val = torch.tensor(val,device = device)

    return train,val
