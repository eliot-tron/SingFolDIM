import torch
import torch.nn as nn


def circle_net(checkpoint_path: str = "", hid_size = [16, 32 ,1, 16, 8], score: bool=False, non_linearity: nn.Module=nn.ReLU(), nclasses=2) -> nn.Module:
    if not isinstance(hid_size, list):
        hid_size = [hid_size]
    net = nn.Sequential(
        nn.Linear(2, hid_size[0]),
        non_linearity,
        *[nn.Sequential(nn.Linear(hid_size[i], hid_size[i + 1]), non_linearity) for i in range(len(hid_size[1:]))],
        nn.Linear(hid_size[-1], nclasses),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net


def shallow_circle_net(checkpoint_path: str = "", hid_size = 16, score: bool=False, non_linearity: nn.Module=nn.ReLU(), nclasses=2) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, nclasses),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net
