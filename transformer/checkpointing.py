import torch

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else "cpu"
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
