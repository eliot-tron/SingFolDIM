from functools import partial
from os import makedirs, path
import time
from pathlib import Path
from typing import Dict, Optional, Union
from torch import nn
import torch
from matplotlib import cm, colors, pyplot as plt
from tqdm import tqdm

from circle_datasets import CircleDataset
from circle_networks import circle_net
from geometry import GeometricModel
from torchvision import datasets, transforms
from torchdiffeq import odeint

import mnist_networks, cifar10_networks
from xor3d_datasets import Xor3dDataset
from xor3d_networks import xor3d_net
from xor_datasets import XorDataset
from xor_networks import xor_net
from autoattack import AutoAttack

# TODO: break into multiple files.

class Experiment(object):

    """Class for storing the values of my experiments. """

    def __init__(self,
                 dataset_name: str,
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int]=None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]]=None,
                 checkpoint_path: Optional[str]=None,
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
        self.dtype = dtype
        self.device = device
        self.num_samples = num_samples
        self.restrict_to_class = restrict_to_class
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
    
    def __str__(self) -> str:
        title = f"{type(self).__name__} object"
        variables = ""
        for key, var in vars(self).items():
            variables += f"- {key}: {var}\n"
        n_dash = (len(title) - len('variables')) // 2
        return title + '\n' + '-' * n_dash + 'variables' + '-' * n_dash + '\n' + variables
    
    def save_info_to_txt(self, save_directory: str):
        saving_path = path.join(save_directory, f"{type(self).__name__}_{self.dataset_name}_info.txt")
        with open(saving_path, 'w') as file:
            file.write(str(self))

    def get_output_dimension(self):
        return self.network(self.input_points[0].unsqueeze(0)).shape[-1]


    def get_input_dimension(self):
        return len(self.input_points[0].flatten())
    
    def get_number_of_classes(self):
        return len(self.input_space['train'].classes)

    def init_geo_model(self):
        """TODO: Docstring for init_geo_model.

        :returns: None

        """
        self.geo_model = GeometricModel(
            network=self.network,
            network_score=self.network_score,
        )

    def init_nl_function(self):
        """Initializes the value of self.nl_function based on the string variable self.non_linearity."""
        if isinstance(self.non_linearity, str):
            if self.non_linearity == 'Sigmoid':
                self.nl_function = nn.Sigmoid()
            elif self.non_linearity == 'ReLU':
                self.nl_function= nn.ReLU()
            elif self.non_linearity == 'GELU':
                if self.dataset_name not in ['XOR', 'MNIST']: print('WARNING: GELU is (for now) only implemented with the weights of the ReLU network.')
                self.nl_function = nn.GELU()

    def init_checkpoint_path(self):
        """Initializes the value of self.checkpoint_path based on self.dataset_name and self.non_linearity."""
        raise NotImplementedError(f"No checkpoint path given for {self.dataset_name}.")

    def init_input_space(self,
                         root: str='data',
                         download: bool=True,
                         ):
        """Initializes the value of self.input_space with a dictionary with two datasets: the train one
        and the test one. It does so based on the 

        :root: Root directory to the dataset if already downloaded.
        :download (bool, optional): If True, downloads the dataset from the internet and
                                    puts it in root directory. If dataset is already downloaded,
                                    it is not downloaded again.

        :returns: None

        """
        if self.input_space is not None:
            if self.restrict_to_class is not None:
                for input_space_train_or_val in self.input_space:
                    restriction_indices = input_space_train_or_val.targets == self.restrict_to_class
                    input_space_train_or_val.targets = input_space_train_or_val.targets[restriction_indices]
                    input_space_train_or_val.data = input_space_train_or_val.data[restriction_indices]
        elif self.dataset_name in ['Noise', 'Adversarial']:
            raise ValueError(f"{self.dataset_name} cannot be a base dataset.")
        else:
            raise NotImplementedError(f"{self.dataset_name} cannot be a base dataset yet.")

    def init_input_points(self, train:bool=True):
        """Initializes the value of self.input_point (torch.Tensor) based on 
        self.input_space, self.num_samples, self.random, and if self.dataset_name
        is Noise if self.adversarial_budget > 0.

        :train: If True, get points from the training dataset,
                else from the validation dataset.

        :returns: TODO

        """
        print(f"Loading {self.num_samples} samples...")
        input_space_train_or_val = self.input_space['train' if train else 'val']

        if self.num_samples > len(input_space_train_or_val):
            print(
                f'WARNING: you are trying to get more samples ({self.num_samples}) than the number of data in the test set ({len(input_space_train_or_val)})')

        if self.random:
            indices = torch.randperm(len(input_space_train_or_val))[:self.num_samples]
        else:
            indices = range(self.num_samples)

        self.input_points = torch.stack([input_space_train_or_val[idx][0] for idx in indices])
        self.input_points = self.input_points.to(self.device).to(self.dtype)
        
        if self.dataset_name == "Noise":
            self.input_points = torch.rand_like(self.input_points).to(self.device).to(self.dtype)
        if self.adversarial_budget > 0:
            print("Computing the adversarial attacks...")
            adversary = AutoAttack(self.network_score, norm='L2', eps=self.adversarial_budget, version='custom', attacks_to_run=['apgd-ce'], device=self.device, verbose=False)
            labels = torch.argmax(self.network_score(self.input_points), dim=-1)
            attacked_points = adversary.run_standard_evaluation(self.input_points.clone(), labels, bs=250)
            self.input_points = attacked_points #.to(self.dtype)
            print("...done!")

    def init_networks(self):
        """Initializes the value of self.network and self.network_score based on the string
        self.dataset_name, and self.pool.

        :arg1: TODO
        :returns: TODO

        """
        if isinstance(self.network, torch.nn.Module) and isinstance(self.network_score, torch.nn.Module):
            self.network = self.network.to(self.device).to(self.dtype)
            self.network_score = self.network_score.to(self.device).to(self.dtype)

            if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
                print(f"Let's use {torch.cuda.device_count()} GPUs!")
                # self.network = nn.DataParallel(self.network)
                # self.network_score = nn.DataParallel(self.network_score)

            print(f"network to {self.device} as {self.dtype} done")
        elif self.dataset_name in ['Noise', 'Adversarial']:
            raise ValueError(f"{self.dataset_name} cannot have an associated network.")
        else:
            raise NotImplementedError(f"The dataset {self.dataset_name} has no associated network yet.")

    def plot_traces(
        self,
        axes,
        output_dir: Union[str, Path]='output/',
        singular_values: bool=False,
        face_color: Optional[str]=None,
        positions: Optional[list[float]]=None,
        box_width: float=1,
    ) -> None:
        """Plot the mean ordered eigenvalues of the Fisher Information Matrix, also called the Local Data Matrix."""

        if not path.isdir(output_dir):
            makedirs(output_dir)
       
        local_data_matrix = self.geo_model.local_data_matrix(self.input_points)

        number_of_batch = local_data_matrix.shape[0]

        if face_color is None:
            face_color = 'white'


        traces = torch.einsum('...ii', local_data_matrix).log10().detach().cpu()

        torch.save(traces, path.join(output_dir, f"experiment_{self.dataset_name}_traces.pt"))

        boxplot = axes.boxplot(traces,
                               positions=positions,
                               widths=box_width,
                               sym=',',
                               patch_artist=True,
                               boxprops=dict(facecolor=face_color),
                               medianprops=dict(color='black'),
                               meanprops=dict(markeredgecolor='black', markerfacecolor=face_color),
                               showmeans=True
                               )
        return boxplot

    def plot_FIM_eigenvalues(
        self,
        axes,
        output_dir: Union[str, Path]='output/',
        singular_values: bool=False,
        known_rank: Optional[int]=None,
        face_color: Optional[str]=None,
        positions: Optional[list[float]]=None,
        box_width: float=1,
    ) -> None:
        """Plot the mean ordered eigenvalues of the Fisher Information Matrix, also called the Local Data Matrix."""

        if not path.isdir(output_dir):
            makedirs(output_dir)
       
        local_data_matrix = self.geo_model.local_data_matrix(self.input_points)

        number_of_batch = local_data_matrix.shape[0]


        if known_rank is None:
            known_rank = min(local_data_matrix.shape[1:])
        else:
            known_rank = min(known_rank, min(local_data_matrix.shape[1:]) - 1)
            
        if face_color is None:
            face_color = 'white'

        if positions is None:
            positions = range(known_rank + 1)

        # TODO: implement a faster computation of the topk eigenvalues <15-04-24> #
        if singular_values:
            eigenvalues = torch.linalg.svdvals(local_data_matrix)
        else:
            with torch.no_grad():
                # t0 = time.time()
                # eigenvalues = torch.linalg.eigvalsh(local_data_matrix) 
                # t1 = time.time()
                try:
                    topk_eigenvalues = torch.lobpcg(local_data_matrix, k=known_rank+1)[0]
                    selected_eigenvalues = topk_eigenvalues.abs().sort(descending=True).values
                except ValueError:
                    eigenvalues = torch.linalg.eigvalsh(local_data_matrix).abs().sort(descending=True).values 
                    selected_eigenvalues = eigenvalues.abs().sort(descending=True).values[...,:known_rank + 1]
            # t2 = time.time()
            # print(f"All: {t1-t0}s, topk: {t2-t1}s.")

        # selected_eigenvalues = eigenvalues.abs().sort(descending=True).values[...,:known_rank + 1]
        # print(f"All close: {torch.allclose(topk_eigenvalues, selected_eigenvalues[...,:known_rank])}")
        #  max_eigenvalues = eigenvalues.max(dim=-1, keepdims=True).values
        #  eigenvalues = eigenvalues / max_eigenvalues
        selected_eigenvalues[selected_eigenvalues < 1e-23] = 1e-23  # to solve problem with log(0)
        oredered_list_eigenvalues = list(selected_eigenvalues.log10().movedim(-1, 0).detach().cpu())  # TODO: log after or before mean? <15-04-24> #

        torch.save(oredered_list_eigenvalues, path.join(output_dir, f"experiment_{self.dataset_name}_orderd_list_eigenvalues.pt"))

        boxplot = axes.boxplot(oredered_list_eigenvalues,
                               positions=positions,
                               widths=box_width,
                               sym=',',
                               patch_artist=True,
                               boxprops=dict(facecolor=face_color),
                               medianprops=dict(color='black'),
                               meanprops=dict(markeredgecolor='black', markerfacecolor=face_color),
                               showmeans=True
                               )
        # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     plt.setp(boxplot[element], color=edge_color)
        
        return boxplot
    
    def plot_foliation(self,
                       transverse: bool=True,
                       nleaves: Optional[int]=None,
                       ) -> None:
        """Plots the kernel / transverse foliation associated to
        the Fisher Information Matrix.

        :transverse: (bool) if True, plot the transverse foliation, else plot the kernel foliation.
        :returns: None

        """
        raise NotImplementedError(f"plot_foliation not implemented for {self.dataset_name} dataset.")

    def batch_compute_leaf(self, init_points, transverse=False):
        """Compute the leaf going through the point
        [init_points] and with distribution the kernel
        of the FIM induced by the network.

        :init_point: point from the leaf with shape (bs, d)
        :num_points: number of points to generate on the curve
        :dt: time interval to solve the PDE system
        :transverse: If true, compute the transverse foliation
        :returns: the curve gamma(t) with shape (bs, n, d)

        """
        if self.geo_model is None:
            self.init_geo_model()
        
        def f(t, y):
            J = self.geo_model.jac_proba(y)
            e = J[:,0,:]
            if not transverse:
                e = torch.stack((e[:,1], -e[:,0])).movedim(0, -1)
            norm = torch.linalg.vector_norm(e, ord=2, dim=-1, keepdim=True)
            e = (e / norm).nan_to_num(0.0)
            return e

        leaf = odeint(f, init_points, t=torch.linspace(0, 0.5 * 4, 100 * 4), method="rk4").transpose(0, 1)
        leaf_back = odeint(f, init_points, t=-torch.linspace(0, 0.5 * 4, 100 * 4), method="rk4").transpose(0, 1)
        
        return torch.cat((leaf_back.flip(1)[:,:-1], leaf), dim=1)


class MNISTExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("MNIST", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        self.checkpoint_path = f'./checkpoint/mnist_medium_cnn_30_{self.pool}_{self.non_linearity}.pt'

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: datasets.MNIST(
            root,
            train=(x=='train'),
            download=download,
            transform=transforms.Compose([transforms.ToTensor()]),
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)

    def init_networks(self):
        maxpool = (self.pool == 'maxpool')
        self.network = mnist_networks.medium_cnn(self.checkpoint_path,
                                                 non_linearity=self.nl_function,
                                                 maxpool=maxpool)
        self.network_score = mnist_networks.medium_cnn(self.checkpoint_path,
                                                       score=True,
                                                       non_linearity=self.nl_function,
                                                       maxpool=maxpool)

        return super().init_networks()


class CIFAR10Exp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("CIFAR10", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        self.checkpoint_path = f'./checkpoint/cifar10_medium_cnn_30_{self.pool}_{self.non_linearity}_vgg11.pt'

    def init_input_space(self, root: str = 'data', download: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.input_space = {x: datasets.CIFAR10(
            root,
            train=(x=='train'),
            download=download,
            transform=transform,
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)

    def init_networks(self):
        maxpool = (self.pool == 'maxpool')
        self.network = cifar10_networks.medium_cnn(self.checkpoint_path, maxpool=maxpool)
        self.network_score = cifar10_networks.medium_cnn(self.checkpoint_path, score=True, maxpool=maxpool)

        return super().init_networks()


class XORExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("XOR", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        if self.non_linearity == 'Sigmoid':
            self.checkpoint_path = './checkpoint/xor_net_sigmoid_20.pt'
        elif self.non_linearity == 'ReLU':
            self.checkpoint_path = './checkpoint/xor_net_relu_30.pt'
        elif self.non_linearity == 'GELU':
            self.checkpoint_path = './checkpoint/xor_net_gelu_acc100.pt'

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: XorDataset(
            nsample=10000,
            test=(x=='val'),
            discrete=False,
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)

    def init_networks(self):
        self.network = xor_net(self.checkpoint_path, non_linearity=self.nl_function)
        self.network_score = xor_net(self.checkpoint_path, score=True, non_linearity=self.nl_function)

        return super().init_networks()

    def plot_foliation(self,
                       transverse: bool=True,
                       nleaves: Optional[int]=None,
                       ) -> None:
        """Plots the kernel / transverse foliation associated to
        the Fisher Information Matrix.

        :transverse: (bool) if True, plot the transverse foliation, else plot the kernel foliation.
        :returns: None

        """
        if nleaves is None:
            nleaves = self.num_samples
        input_space_train = self.input_space['train']
        indices = torch.randperm(len(input_space_train))[:nleaves]
        init_points = torch.stack([input_space_train[idx][0] for idx in indices])
        init_points = init_points.to(self.device).to(self.dtype)
        #  scale = 0.1
        #  xs = torch.arange(0, 1.5 + scale, scale, dtype=self.dtype, device=self.device)
        #  init_points = torch.cartesian_prod(xs, xs)
        print("Plotting the leaves...")
        leaves = self.batch_compute_leaf(init_points, transverse=transverse)

        for leaf in tqdm(leaves):
            plt.plot(leaf[:, 0], leaf[:, 1], color='blue', linewidth=0.2, zorder=1)

        if self.dataset_name == "XOR":
            plt.plot([0, 1], [0, 1], "ro", zorder=3)
            plt.plot([0, 1], [1, 0], "go", zorder=3)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])


class XOR3DExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("XOR3D", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        if self.non_linearity == 'ReLU':
            self.checkpoint_path = './checkpoint/xor3d_net_relu_hl16_acc96.pt'
        else:
            raise NotImplementedError(f"XOR3D not implement for {self.non_linearity}.")

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: Xor3dDataset(
            nsample=10000,
            test=(x=='val'),
            discrete=False,
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)

    def init_networks(self):
        self.network = xor3d_net(self.checkpoint_path, non_linearity=self.nl_function)
        self.network_score = xor3d_net(self.checkpoint_path, score=True, non_linearity=self.nl_function)

        return super().init_networks()

    def plot_foliation(self,
                       transverse: bool=True,
                       nleaves: Optional[int]=None,
                       ) -> None:
        """Plots the kernel / transverse foliation associated to
        the Fisher Information Matrix.

        :transverse: (bool) if True, plot the transverse foliation, else plot the kernel foliation.
        :returns: None

        """
        if nleaves is None:
            nleaves = self.num_samples
        input_space_train = self.input_space['train']
        indices = torch.randperm(len(input_space_train))[:nleaves]
        init_points = torch.stack([input_space_train[idx][0] for idx in indices])
        init_points = init_points.to(self.device).to(self.dtype)
        #  scale = 0.1
        #  xs = torch.arange(0, 1.5 + scale, scale, dtype=self.dtype, device=self.device)
        #  init_points = torch.cartesian_prod(xs, xs)
        print("Plotting the leaves...")
        leaves = self.batch_compute_leaf(init_points, transverse=transverse)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for leaf in tqdm(leaves):
            if transverse:
                ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2], color='blue', linewidth=0.2, zorder=1)
            else:
                X, Y = torch.meshgrid(leaf[:, 0], leaf[:, 1])
                Z = leaf[:, 2].unsqueeze(0).expand(leaf.shape[0], -1)
                ax.plot_wireframe(X, Y, Z, color='blue', zorder=1, rcount=10, ccount=10)

        for (inp, label) in Xor3dDataset(test=True, discrete=True, nsample=8):
            if label == 1:
                ax.plot(inp[0], inp[1], inp[2], "go", zorder=3)
            if label == 0:
                ax.plot(inp[0], inp[1], inp[2], "ro", zorder=3)
        ax.axes.set_xlim3d(-0.1, 1.1)
        ax.axes.set_ylim3d(-0.1, 1.1)
        ax.axes.set_zlim3d(-0.1, 1.1)
        #  plt.show()


class CircleExp(Experiment):
    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None,
                 nclasses: int=2,
                 ):
        self.nclasses = nclasses
        super().__init__(f"Circle{nclasses}", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        self.checkpoint_path = f'./checkpoint/deep_circle_net_c{self.nclasses}_{self.non_linearity.lower()}_30.pt'

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: CircleDataset(
            nsample=10000,
            test=(x=='val'),
            nclasses=self.nclasses,
            noise=True,
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)

    def init_networks(self):
        self.network = circle_net(self.checkpoint_path, non_linearity=self.nl_function, nclasses=self.nclasses)
        self.network_score = circle_net(self.checkpoint_path, score=True, non_linearity=self.nl_function, nclasses=self.nclasses)

        return super().init_networks()

    def plot_foliation(self,
                       transverse: bool=True,
                       nleaves: Optional[int]=None,
                       ) -> None:
        """Plots the kernel / transverse foliation associated to
        the Fisher Information Matrix.

        :transverse: (bool) if True, plot the transverse foliation, else plot the kernel foliation.
        :returns: None

        """
        if nleaves is None:
            nleaves = self.num_samples
        input_space_train = self.input_space['train']
        # indices = torch.randperm(len(input_space_train))[:nleaves]
        indices = torch.linspace(0, len(input_space_train), nleaves + 1).int()[:-1]
        data_points = torch.stack([input_space_train[idx][0] for idx in indices])
        data_classes = torch.stack([input_space_train[idx][1] for idx in indices])
        init_points = torch.rand_like(data_points) * 2 - 1
        # init_points = data_points
        init_points = init_points.to(self.device).to(self.dtype)
        #  scale = 0.1
        #  xs = torch.arange(0, 1.5 + scale, scale, dtype=self.dtype, device=self.device)
        #  init_points = torch.cartesian_prod(xs, xs)
        print("Plotting the leaves...")
        leaves = self.batch_compute_leaf(init_points, transverse=transverse)

        for leaf in tqdm(leaves.cpu()):
            plt.plot(leaf[:, 0], leaf[:, 1], color='blue', linewidth=0.2, zorder=1)

        print("...plotting the data points...")
        cmap = cm.get_cmap('jet', self.nclasses)  
        # plt.colorbar(ticks=range(self.nclasses + 2))
        scamap = cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=self.nclasses-0.5), cmap=cmap)
        # plt.colorbar(scamap, ticks=range(self.nclasses))
        for p, c in zip(data_points, data_classes):
            plt.plot(p[0], p[1], "o", color=scamap.to_rgba(c))

        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])


class LettersExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("Letters", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: datasets.EMNIST(
            root,
            train=(x=='train'),
            download=download,
            split="letters",
            transform=transforms.Compose([transforms.ToTensor()]),
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)


class FashionMNISTExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("FashionMNIST", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: datasets.FashionMNIST(
            root,
            train=(x=='train'),
            download=download,
            transform=transforms.Compose([transforms.ToTensor()]),
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)


class KMNISTExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("KMNIST", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: datasets.KMNIST(
            root,
            train=(x=='train'),
            download=download,
            transform=transforms.Compose([transforms.ToTensor()]),
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)


class QMNISTExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("QMNIST", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_input_space(self, root: str = 'data', download: bool = True):
        self.input_space = {x: datasets.QMNIST(
            root,
            train=(x=='train'),
            download=download,
            transform=transforms.Compose([transforms.ToTensor()]),
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)


class CIFARMNISTExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("CIFARMNIST", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_input_space(self, root: str = 'data', download: bool = True):
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Resize(size=(28, 28)),
            ])

        self.input_space = {x: datasets.CIFAR10(
            root,
            train=(x=='train'),
            download=download,
            transform=transform,
        ) for x in ['train', 'val']
        }
        return super().init_input_space(root, download)


# TODO: what to do with Noise and Adversarial? function ? class ? how to do not-base-experiments ? Remove adversarial_budget and Noise/Adversarial from dataset_name

class NoiseExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("Noise", 
                         non_linearity,
                         adversarial_budget,
                         dtype,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network,
                         network_score)

    def init_checkpoint_path(self):
        raise ValueError(f"{self.dataset_name} cannot have an associated network.")

    def init_input_space(self, root: str = 'data', download: bool = True):
        raise ValueError(f"{self.dataset_name} cannot be a base dataset.")

    def init_input_points(self, train:bool=True):
        if self.input_space is not None:
            super().init_input_points(train=train)
            self.input_points = torch.rand_like(self.input_points).to(self.device).to(self.dtype)
        else:
            raise ValueError(f"{self.dataset_name} cannot be a base dataset.")

    def init_networks(self):
        raise ValueError(f"{self.dataset_name} cannot have an associated network.")


class AdversarialExp(Experiment):

    def __init__(self, 
                 non_linearity: str,
                 adversarial_budget: float,
                 dtype: torch.dtype,
                 device: torch.DeviceObjType,
                 num_samples: int,
                 pool: str,
                 random: bool,
                 restrict_to_class: Optional[int] = None,
                 input_space: Optional[Dict[str, datasets.VisionDataset]] = None,
                 checkpoint_path: Optional[str] = None,
                 network: Optional[nn.Module] = None,
                 network_score: Optional[nn.Module] = None):
        super().__init__("Adversarial", 
                         non_linearity,
                         adversarial_budget,
                         torch.float,
                         device,
                         num_samples,
                         pool,
                         random,
                         restrict_to_class,
                         input_space,
                         checkpoint_path,
                         network.float(),
                         network_score.float())
        if dtype != torch.float:
            print("WARNING: Adversarial attack only implemented for float32 due to external dependences.")

    def init_checkpoint_path(self):
        raise ValueError(f"{self.dataset_name} cannot have an associated network.")

    def init_input_space(self, root: str = 'data', download: bool = True):
        raise ValueError(f"{self.dataset_name} cannot be a base dataset.")

    def init_input_points(self, train:bool=True):
        if self.input_space is not None:
            super().init_input_points(train=train)
        else:
            raise ValueError(f"{self.dataset_name} cannot be a base dataset.")
        # if self.adversarial_budget > 0:
        #     print("Computing the adversarial attacks...")
        #     adversary = AutoAttack(self.network_score.float(), norm='L2', eps=self.adversarial_budget, version='custom', attacks_to_run=['apgd-ce'], device=self.device, verbose=False)
        #     labels = torch.argmax(self.network_score(self.input_points.float()), dim=-1)
        #     attacked_points = adversary.run_standard_evaluation(self.input_points.clone().float(), labels, bs=250)
        #     self.input_points = attacked_points.to(self.dtype)
        #     print("...done!")
        # else:
        #     raise ValueError("Adversarial dataset but with self.adversarial_budget <= 0.")

    def init_networks(self):
        raise ValueError(f"{self.dataset_name} cannot have an associated network.")


implemented_experiment_dict = {
    "MNIST": MNISTExp,
    "CIFAR10": CIFAR10Exp,
    "XOR": XORExp,
    "XOR3D": XOR3DExp,
    "Letters": LettersExp,
    "KMNIST": KMNISTExp,
    "QMNIST": QMNISTExp,
    "CIFARMNIST": CIFARMNISTExp,
    "FashionMNIST": FashionMNISTExp,
    "Noise": NoiseExp,
    "Adversarial": AdversarialExp,
    "Circle2": partial(CircleExp, nclasses=2),
    "Circle3": partial(CircleExp, nclasses=3),
    "Circle6": partial(CircleExp, nclasses=6),
}
