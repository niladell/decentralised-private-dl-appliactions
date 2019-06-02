import torch
import torchvision
from torchvision import datasets, transforms
import syft as sy
import logging

logger = logging.getLogger(__name__)

use_cuda = True # TODO This requires to be present in every file and it's obvioulsy a pain in the ass
                # should find a way to fix it properly
if not torch.cuda.is_available() and use_cuda:
        use_cuda = False
        logger.warning('Cuda unabailable, falling back to CPU')

device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def get_datasets(dataset_name='CIFAR'): # TODO Pass argumetn of dataset
    dataset_list = {
        'CIFAR': datasets.CIFAR10,  # CIFAR defaults to CIFAR10
        'CIFAR10': datasets.CIFAR10,
        'MNIST': datasets.MNIST,
        'FMNIST': datasets.FashionMNIST
    }
    dataset = dataset_list[dataset_name]

    train_dataset = dataset(
        '../data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
        ]))

    test_dataset = dataset(
        '../data', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
    return train_dataset, test_dataset

def get_dataloaders(batch_size: int, federate_workers: list = None, **kwargs):

    train_dataset, test_dataset = get_datasets()

    if federate_workers is None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    else:

        train_loader = sy.FederatedDataLoader(
            train_dataset.federate(federate_workers),  #pylint: disable=no-member
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader