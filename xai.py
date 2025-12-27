import torch

def node_importance(model, data):
    model.eval()
    data = data.clone()
    data.x = data.x.float()
    data.x.requires_grad_(True)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    out = model(data)
    out.backward()
    importance = data.x.grad.abs().sum(dim=1)

    return importance.detach().cpu()
