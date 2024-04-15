from os import makedirs, path
from pathlib import Path
from typing import Optional, Union
from torch import nn
import torch
from matplotlib import pyplot as plt

from geometry import GeometricModel
from torchvision import datasets, transforms

import mnist_networks, cifar10_networks
from xor_datasets import XorDataset
from xor_networks import xor_net
from autoattack import AutoAttack


class Experiment(object):

    """Class for storing the values of my experiments. """

    def __init__(self,
                 dataset_name: str,
                 non_linearity: str,
                 adversarial_budget: float,
                 precision_type: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 noise: bool,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int]=None,
                 checkpoint_path: Optional[str]=None,
                 input_space: Optional[datasets.VisionDataset]=None,
                 network: Optional[nn.Module]=None,
                 network_score: Optional[nn.Module]=None,
                 ):
        """TODO: to be defined. """
        self.dataset_name = dataset_name
        self.network = network 
        self.network_score = network_score
        self.non_linearity = non_linearity
        self.checkpoint_path = checkpoint_path
        self.adversarial_budget = adversarial_budget
        self.precision_type = precision_type
        self.device = device
        self.num_samples = num_samples
        self.restrict_to_class = restrict_to_class
        self.noise = noise
        self.pool = pool
        self.input_space = input_space
        self.random = random

        self.geo_model = None
        self.nl_function = None  # typing: ignore

        self.init_nl_function()
        if self.checkpoint_path is None:
            self.init_checkpoint_path()
        if self.network is None or self.network_score is None:
            self.init_networks()
        if self.input_space is None:
            self.init_input_space()
        self.init_input_points()
        self.init_geo_model()


    def init_geo_model(self):
        """TODO: Docstring for init_geo_model.

        :returns: None

        """
        self.geo_model = GeometricModel(
            network=self.network,
            network_score=self.network_score,
        )

    def init_nl_function(self):
        if isinstance(self.non_linearity, str):
            if self.non_linearity == 'Sigmoid':
                self.nl_function = nn.Sigmoid()
            elif self.non_linearity == 'ReLU':
                self.nl_function= nn.ReLU()
            elif self.non_linearity == 'GELU':
                print('WARNING: GELU is (for now) only implemented with the weights of the ReLU network.')
                self.nl_function = nn.GELU()

    def init_checkpoint_path(self):
        if self.dataset_name == "MNIST":
            if self.non_linearity == 'Sigmoid':
                self.checkpoint_path = f'./checkpoint/mnist_medium_cnn_30_{self.pool}_Sigmoid.pt'
            elif self.non_linearity == 'ReLU':
                self.checkpoint_path = f'./checkpoint/mnist_medium_cnn_30_{self.pool}_ReLU.pt'
            elif self.non_linearity == 'GELU':
                self.checkpoint_path = f'./checkpoint/mnist_medium_cnn_10_{self.pool}_ReLU.pt'
        elif self.dataset_name == 'CIFAR10':
            if self.non_linearity == 'Sigmoid':
                raise NotImplementedError
            elif self.non_linearity == 'ReLU':
                self.checkpoint_path = f'./checkpoint/cifar10_medium_cnn_30_{self.pool}_ReLU_vgg11.pt'
            elif self.non_linearity == 'GELU':
                self.checkpoint_path = f'./checkpoint/cifar10_medium_cnn_30_{self.pool}_ReLU_vgg11.pt'
        elif self.dataset_name == "XOR":
            if self.non_linearity == 'Sigmoid':
                self.checkpoint_path = './checkpoint/xor_net_sigmoid_20.pt'
            elif self.non_linearity == 'ReLU':
                self.checkpoint_path = './checkpoint/xor_net_relu_30.pt'
            elif self.non_linearity == 'GELU':
                self.checkpoint_path = './checkpoint/xor_net_relu_30.pt'

    def init_input_space(self,
                         root: str='data',
                         download: bool=True,
                         train=False,
                         ):
        """TODO: Docstring for init_input_space.

        :root: Root directory to the dataset if already downloaded.
        :train (bool, optional): If True, creates dataset from training.pt, 
            otherwise from test.pt.
        :download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
    downloaded again.

        :returns: None

        """
        if self.dataset_name == 'MNIST':
            self.input_space = datasets.MNIST(
                root,
                train=train,
                download=download,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        elif self.dataset_name == 'EMNIST-letters':
            self.input_space = datasets.EMNIST(
                root,
                train=train,
                download=download,
                split="letters",
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

            self.input_space = datasets.CIFAR10(
                root,
                train=train,
                download=download,
                transform=transform,
            )
        elif self.dataset_name == 'XOR':
            input_space = XorDataset(
                nsample=10000,
                discrete=False,
            )
        elif self.dataset_name == 'Noise':
            raise NotImplementedError("Noise cannot be a base dataset yet.")
        elif self.dataset_name == 'Adversarial':
            raise ValueError("Adversarial cannot be a base dataset.")

        if self.restrict_to_class is not None:
            restriction_indices = self.input_space.targets == self.restrict_to_class
            self.input_space.targets = self.input_space.targets[restriction_indices]
            self.input_space.data = self.input_space.data[restriction_indices]

    def init_input_points(self):
        """TODO: Docstring for init_input_points.

        :returns: TODO

        """
        print(f"Loading {self.num_samples} samples...")

        if self.num_samples > len(self.input_space):
            print(
                f'WARNING: you are trying to get more samples ({self.num_samples}) than the number of data in the test set ({len(self.input_space)})')

        if self.random:
            indices = torch.randperm(len(self.input_space))[:self.num_samples]
        else:
            indices = range(self.num_samples)

        self.input_points = torch.stack([self.input_space[idx][0] for idx in indices])
        self.input_points = self.input_points.to(self.device).to(self.precision_type)
        
        if self.noise:
            self.input_points = torch.cat([self.input_points, torch.rand_like(self.input_points).to(self.device).to(self.precision_type)], dim=0)
        if self.adversarial_budget > 0:
            print("Computing the adversarial attacks...")
            adversary = AutoAttack(self.network_score.float(), norm='L2', eps=self.adversarial_budget, version='custom', attacks_to_run=['apgd-ce'], device=self.device, verbose=False)
            labels = torch.argmax(self.network_score(self.input_points), dim=-1)
            attacked_points = adversary.run_standard_evaluation(self.input_points.clone().float(), labels, bs=250)
            self.input_points = attacked_points.to(self.precision_type)
            print("...done!")



    def init_networks(self):
        """TODO: Docstring for init_networks.

        :arg1: TODO
        :returns: TODO

        """
        maxpool = (self.pool == 'maxpool')
        if self.dataset_name in ['MNIST', 'EMNIST']:
            self.network = mnist_networks.medium_cnn(self.checkpoint_path, non_linearity=self.nl_function, maxpool=maxpool)
            self.network_score = mnist_networks.medium_cnn(
                self.checkpoint_path, score=True, non_linearity=self.nl_function, maxpool=maxpool)
        elif self.dataset_name == 'CIFAR10':
            self.network = cifar10_networks.medium_cnn(self.checkpoint_path, maxpool=maxpool)
            self.network_score = cifar10_networks.medium_cnn(self.checkpoint_path, score=True, maxpool=maxpool)
        elif self.dataset_name == 'XOR':
            self.network = xor_net(self.checkpoint_path, non_linearity=self.nl_function)
            self.network_score = xor_net(self.checkpoint_path, score=True, non_linearity=self.nl_function)
        else:
            raise NotImplementedError(f"The dataset {self.dataset_name} has no associated network yet.")

        self.network = self.network.to(self.device).to(self.precision_type)
        self.network_score = self.network_score.to(self.device).to(self.precision_type)

        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.network = nn.DataParallel(self.network)
            self.network_score = nn.DataParallel(self.network_score)

        print(f"network to {self.device} as {self.precision_type} done")


    def plot_FIM_eigenvalues(
        self,
        axes,
        output_dir: Union[str, Path]='output/',
        output_name: Optional[str]=None,
        singular_values: bool=False,
        known_rank: Optional[int]=None,
        edge_color: Optional[str]=None,
    ) -> None:
        """Plot the mean ordered eigenvalues of the Fisher Information Matrix, also called the Local Data Matrix."""

        if not path.isdir(output_dir):
            makedirs(output_dir)
       
        local_data_matrix = self.geo_model.local_data_matrix(self.input_points)

        number_of_batch = local_data_matrix.shape[0]

        if known_rank is None:
            known_rank = min(local_data_matrix.shape[1:])

        # TODO: implement a faster computation of the topk eigenvalues <15-04-24, eliot> #
        if singular_values:
            eigenvalues = torch.linalg.svdvals(local_data_matrix)
        else:
            eigenvalues = torch.linalg.eigvalsh(local_data_matrix) 
    #  max_eigenvalues = eigenvalues.max(dim=-1, keepdims=True).values
    #  eigenvalues = eigenvalues / max_eigenvalues
        #  oredered_list_eigenvalues = list(eigenvalues.abs().sort(descending=True).values[...,:known_rank + 1].movedim(-1, 0).detach()) 
        oredered_list_eigenvalues = list(eigenvalues.abs().sort(descending=True).values[...,:known_rank + 1].log10().movedim(-1, 0).detach())  # TODO: log after or before mean? <15-04-24, eliot> #

        boxplot = axes.boxplot(oredered_list_eigenvalues,)
                               #  boxprops=dict(facecolor=edge_color, color=edge_color),
                                #  capprops=dict(color=edge_color),
                                #  whiskerprops=dict(color=edge_color),
                                #  flierprops=dict(color=edge_color, markeredgecolor=edge_color),
                                #  medianprops=dict(color=edge_color),
                               #  )
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(boxplot[element], color=edge_color)

#  def box_plot(data, edge_color, fill_color):
    #  bp = ax.boxplot(data, patch_artist=True)
    
    #  for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #  plt.setp(bp[element], color=edge_color)

    #  for patch in bp['boxes']:
        #  patch.set(facecolor=fill_color)       
        
    #  return bp
