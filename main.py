
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from torchvision import datasets, transforms
import syft as sy


from src.core import train, test
from src.models import CIFAR

logger = logging.getLogger(__name__)

use_cuda = True
if not torch.cuda.is_available() and use_cuda:
        use_cuda = False
        logger.warning('Cuda unabailable, falling back to CPU')
if use_cuda:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = torch.device("cuda" if use_cuda else "cpu")
print(device)
batch_size = 32
epochs = 30
log_iteration = 1000

traindataloader, testdataloader = CIFAR.get_dataloaders(batch_size)
net = CIFAR.CifarNet().to(device)

## Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# training_time = time.time()
# train(model=net,
#       dataloader=traindataloader,
#       criterion=criterion,
#       optimizer=optimizer,
#       device=device,
#       epochs=epochs, log_iteration=log_iteration)
# training_time = time.time() - training_time
#
# _, accuracy = test(net, testdataloader, device)
#
# print(f' --- > Test accuracy: {accuracy} -- Training time {training_time}')

#----- FEDERATE FROM HERE

from src.core import create_virtual_workers

workers, _ = create_virtual_workers(5)

traindataloader, testdataloader = CIFAR.get_dataloaders(batch_size, federate_workers=workers)
net = CIFAR.CifarNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

from src.core import train_federate_simple

training_time = time.time()
train_federate_simple(model=net,
      dataloader=traindataloader,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      epochs=epochs,log_iteration=log_iteration)
training_time = time.time() - training_time

_, accuracy = test(net, testdataloader, device)

print(f' --- > Test accuracy: {accuracy} -- Training time {training_time}')
