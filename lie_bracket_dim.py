import argparse
from copy import deepcopy
from datetime import datetime
from functools import partial
from os import makedirs, path
import random
import time
from typing import Callable, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange
from geometry import GeometricModel
from experiment import Experiment

import mnist_networks, cifar10_networks

from torchvision import datasets, transforms
from xor_datasets import XorDataset
from xor_networks import xor_net
import plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the involutivity of the distribution given by the gradients of the neural network.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default="MNIST",
        choices=['MNIST', 'Letters', 'FashionMNIST', 'KMNIST', 'QMNIST', 'CIFARMNIST', 'XOR', 'XOR3D', 'CIFAR10'],
        metavar='name',
        help="Dataset name to be used.",
    )
    parser.add_argument(
        "--restrict",
        type=int,
        metavar="class",
        default=None,
        help="Class to restrict the main dataset to if needed.",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        metavar='N',
        default=2,
        help="Number of initial points to consider."
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Permutes randomly the inputs."
    )

    parser.add_argument(
        "--savedirectory",
        type=str,
        metavar='path',
        default='./output/',
        help="Path to the directory to save the outputs in."
    )

    parser.add_argument(
        "--maxpool",
        action="store_true",
        help="Use the legacy architecture with maxpool2D instead of avgpool2d."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force device to cpu."
    )
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        nargs='+',
        default="ReLU",
        choices=['Sigmoid', 'ReLU', 'GELU'],
        help="Non linearity used by the network."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        device = torch.device('cpu')
    print(f"Device: {device}")

    dataset_names = args.datasets
    num_samples = args.nsample
    non_linearities =  args.nl
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names] * len(non_linearities)
    elif len(dataset_names) == 1:
        dataset_names = dataset_names * len(non_linearities)
    if not isinstance(non_linearities, list):
        non_linearities = [non_linearities] * len(dataset_names)
    elif len(non_linearities) == 1:
        non_linearities = non_linearities * len(dataset_names)
    dtype = torch.double
    restrict_to_class = None

    pool = "maxpool" if args.maxpool else "avgpool"
    date = datetime.now().strftime("%y%m%d-%H%M%S")
    savedirectory = args.savedirectory + \
        ("" if args.savedirectory[-1] == '/' else '/') + \
        f"{'-'.join(dataset_names)}/involutivity_check/{dtype}/" + \
        f"{date}_nsample={num_samples}{f'_class={restrict_to_class}' if restrict_to_class is not None else ''}_{pool}_{'-'.join(non_linearities)}/"
    if not path.isdir(savedirectory):
        makedirs(savedirectory)

    if not args.random:
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.backends.cudnn.deterministic = True # type: ignore

    experiment_list = []
    for (dataset, non_linearity) in zip(dataset_names, non_linearities):
        print(dataset, non_linearity)
        experiment = Experiment(
            dataset_name=dataset,
            non_linearity=non_linearity,
            adversarial_budget=0,
            dtype=dtype,
            device=device,
            num_samples=num_samples,
            restrict_to_class=restrict_to_class,
            pool=pool,
            random=args.random,
        )
        experiment_list.append(experiment)

    #  normalize = transforms.Normalize((0.,), (1.,))
    random_noise = True

    for i, experiment in enumerate(tqdm(experiment_list)):
        print(f"Testing experiment n°{i}: {experiment.dataset_name} - {experiment.non_linearity}.")
        geo_model = experiment.geo_model
        input_points = deepcopy(experiment.input_points)
        if random_noise:
            input_points = torch.rand_like(input_points).to(device).to(dtype)
        jac = geo_model.jac_proba(input_points) # (..., a, l) 
        lie_bracket = geo_model.lie_bracket(input_points) # (..., b, c, l)
        lie_bracket = lie_bracket.flatten(-3, -2)

        rank_jac = torch.linalg.matrix_rank(jac, atol=None)

        print(f"Rank of the distribution: {rank_jac} (mean={rank_jac.float().mean()}).")

        rank_lie_bracket = torch.linalg.matrix_rank(lie_bracket, atol=None)
        print(f"Rank of the lie brackets: {rank_lie_bracket} (mean={rank_lie_bracket.float().mean()}).")

        cat = torch.cat((jac, lie_bracket), dim=-2)
        print(f"Dimension after concatenation: {cat.shape}.")
        rank_cat = torch.linalg.matrix_rank(cat, atol=None)
        print(f"Rank of the sum: {rank_cat} (mean={rank_cat.float().mean()}).")
        torch.save((rank_jac, rank_lie_bracket, rank_cat),  savedirectory +  f"rank_jac_rank_lie_bracket_rank_cat_{experiment.dataset_name}_{experiment.non_linearity}.pt")



