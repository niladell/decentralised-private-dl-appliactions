import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)

import syft as sy
hook = sy.TorchHook(torch)

def create_virtual_workers(num_workers = None, id_list = None):
    if id_list is None:
        if num_workers is not None:
            id_list = [f'worker_{i}' for i in range(num_workers)]
    if num_workers is not None and len(id_list) != num_workers:
        logger.warning('Number of workers and id_list length not maching.'
                       ' id_list will be taken and num_workers will be ignored.')

    workers = [sy.VirtualWorker(hook, id=i) for i in id_list]
    return workers, id_list