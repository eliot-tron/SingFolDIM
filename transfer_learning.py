from copy import deepcopy
import os
import time
from typing import Dict
from torch import nn
import torch
from torchvision import datasets

from experiment import Experiment


class TransferLearning(object):
    
    """Class for doing transfer learning from one model train on a dataset
    to another dataset. Also includes some performance metrics.
    """

    def __init__(self,
                 base_experiment: Experiment, # TODO: use args <18-04-24> #
                 target_experiment: Experiment,
                 ):
        """Initialize the transfer learning class.

        Args:
            base_model (nn.Module): logit model.
            target_datasets (Dict[datasets.VisionDataset]): target datasets ('train' and 'val') for the new task.
        """
        self.base_experiment = base_experiment
        self.target_experiment = target_experiment
        self.base_model = base_experiment.network
        self.target_datasets = target_experiment.input_space
        self.number_of_class = len(self.target_datasets['train'].classes)
        self.new_model = None
        self.index_new_layer = None
    
    def __str__(self) -> str:
        title = "TransferLearning object"
        variables = ""
        for key, var in vars(self).items():
            variables += f"- {key}: {var}\n"
        n_dash = (len(title) - len('variables')) // 2
        return title + '\n' + '-' * n_dash + 'variables' + '-' * n_dash + '\n' + variables

    
    def save_info_to_txt(self, save_directory: str):
        saving_path = os.path.join(save_directory, f"experiment_base={self.base_experiment.dataset_name}_target={self.target_experiment.dataset_name}_info.txt")
        with open(saving_path, 'w') as file:
            file.write(str(self))


    def init_new_model(
        self,
        fix_other_layers: bool=True
    ) -> None: 
        self.new_model = deepcopy(self.base_model)

        # if isinstance(self.new_model, nn.DataParallel):
        #     model = self.new_model.module
        # else:
        #     model = self.new_model

        if isinstance(self.new_model, nn.Sequential):
            index_last_linear_layer = None
            for i, layer in enumerate(self.new_model):
                if fix_other_layers:
                    layer.requires_grad = False
                if isinstance(layer, nn.Linear):
                    index_last_linear_layer = i
            last_linear_layer = self.new_model[index_last_linear_layer]
            self.new_model[index_last_linear_layer] = nn.Linear(
                in_features=last_linear_layer.in_features,
                out_features=self.number_of_class,
                bias=last_linear_layer.bias is not None,
                device=last_linear_layer.weight.device,
                dtype=last_linear_layer.weight.dtype,
            )
        else:
            raise NotImplementedError(f"TransferLearning only implemented for torch.nn.Sequential, and not for \n{self.new_model}")
        
        # TODO: device + dtype
        self.index_new_layer = index_last_linear_layer

    def train_new_model(
        self,
        fix_other_layers: bool=True,
        output_dir: str='./checkpoint/',
        num_epochs: int=30,
        batch_size: int=4,
        num_workers: int=4,
        lr_start: float=0.001,
    ) -> nn.Module:
        
        if self.new_model is None:
            self.init_new_model(fix_other_layers=fix_other_layers)

        # if isinstance(self.new_model, nn.DataParallel): #TODO: do better
        #     model = self.new_model.module
        # else:
        #     model = self.new_model

        if isinstance(self.new_model, nn.Sequential) and isinstance(self.new_model[-1], nn.LogSoftmax):
            criterion = nn.NLLLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer_function = lambda parameters: torch.optim.SGD(parameters, lr=lr_start, momentum=0.9)
        if fix_other_layers:
            optimizer = optimizer_function(self.new_model[self.index_new_layer].parameters())
        else:
            optimizer = optimizer_function(self.new_model.parameters())

        since = time.time()

        output_dir = os.path.join(output_dir, f"base={self.base_experiment.dataset_name}_target={self.target_experiment.dataset_name}/")
        os.makedirs(output_dir, exist_ok=True)

        suffix_name = f"{'_fixed' if fix_other_layers else ''}_bs={batch_size}_workers={num_workers}_lr={lr_start}"  # TODO: do a txt file and a unique id and good output_dir with the time and all
        best_model_params_path = os.path.join(output_dir, f"best_model_params_by_transfer_learning{suffix_name}.pt")
        metrics_path = os.path.join(output_dir, f"loss_and_acc_during_transfer_learning{suffix_name}.pt")

        torch.save(self.new_model.state_dict(), best_model_params_path)
        best_acc = 0.0

        dataloaders = {x: torch.utils.data.DataLoader(self.target_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers) for x in ['train', 'val']}
        dataset_sizes = {x: len(self.target_datasets[x]) for x in ['train', 'val']}

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = next(self.new_model.parameters()).device
        dtype = next(self.new_model.parameters()).dtype

        loss_list = {x: [] for x in ['train', 'val']}
        acc_list = {x: [] for x in ['train', 'val']}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.new_model.train()  # Set model to training mode
                else:
                    self.new_model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data in batches.
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    inputs = inputs.to(dtype)
                    targets = targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.new_model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == targets.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                loss_list[phase].append(epoch_loss)
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                acc_list[phase].append(epoch_acc.cpu())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.new_model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        torch.save((loss_list, acc_list), metrics_path)

        # load best model weights
        self.new_model.load_state_dict(torch.load(best_model_params_path))
        return self.new_model, loss_list, acc_list
