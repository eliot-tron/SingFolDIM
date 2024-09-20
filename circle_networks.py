import torch
import torch.nn as nn


def circle_net(checkpoint_path: str = "", hid_size = 16, score: bool=False, non_linearity: nn.Module=nn.ReLU(), nclasses=2) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, nclasses),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net
