import torch
from torch.utils import data

# dataset generator
class CircleDataset(data.Dataset):
    """Circle dataset with n classes."""
    
    
    def __init__(self, nsample=1000, test=False, nclasses=2, noise=False):
        """Init the dataset with [nclasses] classes."""
        
        data.Dataset.__init__(self)
        self.nsample = (nsample // nclasses + 1) * nclasses  # make it a multiple of nclasses
        print(f"Warning: you asked for {nsample} samples with {nclasses} classes.\nTo have the same amount of samples per class, the new number of samples is {self.nsample}.")
        self.test = test
        self.nclasses = nclasses
        # if test:
        #     self.nsample //= 10
        t = [(torch.rand((self.nsample // nclasses)) + k) / nclasses for k in range(nclasses)]
        self.input_vars = torch.cat([
            torch.stack(
                (torch.cos(2 * torch.pi * t_k),
                 torch.sin(2 * torch.pi * t_k)),
                dim=-1)
            for t_k in t], dim=0)

        if noise:
            for i, p in enumerate(self.input_vars):
                self.input_vars[i] = p * (torch.randn(1) * 0.01 + 1)
    

    def __getitem__(self, index):
        """Get a data point."""
        assert index <= self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, index // (self.nsample // self.nclasses)

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample

# CD = CircleDataset(100, nclasses=6, noise=True)
# import matplotlib.pyplot as plt
# colors = ['red', 'blue', 'yellow', 'black', 'green', 'orange']
# print(len(CD))
# for p, c in CD:
#     plt.plot(p[0], p[1], "o" ,color=colors[c])
#
# plt.show()
