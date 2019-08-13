import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from utils_basic_mi_example import AlexNet, mlleaks_mlp, weights_init

import logging
import logging
logFormatter ='%(asctime)s %(levelname)s -- %(name)s {%(module)s} [%(funcName)s]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logFormatter, datefmt='%m-%d,%H:%M:%S')

logger = logging.getLogger(__name__)

logger.info("Python: %s" % sys.version)
logger.info("Pytorch: %s" % torch.__version__)

# determine device to run network on (runs on gpu if available)
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_epochs = 100
attack_epochs = 10
batch_size = 128
k = 100
mi_split = 0.2
# lr = 0.001
num_workers=4

# dataset_to_use = 'CIFAR10'
# model_to_use = 'vgg16'

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--optimizer_set', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--data_aug', type=str, default='false')
parser.add_argument('--train_scheduler', type=str, default='false')
parser.add_argument('--clean_start', type=str, default='true')
args = parser.parse_args()
logger.info(args)

n_epochs = args.epochs


clean_start = args.clean_start.lower() == 'true' 
data_aug = args.data_aug.lower() == 'true' 
train_scheduler = args.train_scheduler.lower() == 'true'
optimizer_set = args.optimizer_set
seed_set = args.seed
model_to_use = args.model
dataset_to_use = args.dataset
lr = args.lr

if model_to_use == 'vanilla':
    target_net_type = models.mlleaks_cnn
elif model_to_use == 'alexnet':
    target_net_type = lambda: torchvision.models.alexnet(num_classes=100)
elif model_to_use == 'vgg16':
    import vgg_100
    target_net_type = lambda: vgg_100.vgg16_bn()
elif model_to_use == 'resnet-32':
    import resnet
    target_net_type = lambda: resnet.resnet32(num_classes=100)
    resnet.test(target_net_type())
elif model_to_use == 'resnet-34':
    import resnet_new as res
    target_net_type = res.resnet34
elif model_to_use == 'inception_v3':
    target_net_type = torchvision.models.inception_v3
else:
    raise(NotImplementedError)


def get_Imagenet(*args, **kwargs):
    if kwargs['train']:
        kwargs['split'] = 'train'
    else:
        kwargs['split'] = 'val'
    del kwargs['train']
    return torchvision.datasets.ImageNet(*args, **kwargs)

def get_SVHN(*args, **kwargs):
    if kwargs['train']:
        kwargs['split'] = 'train'
    else:
        kwargs['split'] = 'test'
    del kwargs['train']
    return torchvision.datasets.SVHN(*args, **kwargs)


dataset_list = {
    'CIFAR10': (torchvision.datasets.CIFAR10, 10),
    'CIFAR100':  (torchvision.datasets.CIFAR100, 100),
    'SVHN': (get_SVHN, 10),
    'ImageNet': get_Imagenet
}

dataset, num_classes = dataset_list[dataset_to_use]


if data_aug:
    train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])
    dataset_to_use = f'{dataset_to_use}-wAug'
else:
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
])


# ('sgd', optim.SGD), , ('adadelta', optim.Adadelta, {}),
# optimizer_list = [  ('rmsprop', optim.RMSprop, {'eps': 1e-5})]
# optimizer_list = [('adam', optim.Adam, {'eps': 1e-5})]
# optimizer_list = [('sgd', optim.SGD, {})]


optimizer_list = [('sgd', optim.SGD, {}),
                  ('sgd_momentum', optim.SGD, {'momentum': 0.9}),
                  ('sgd_momentum_wd', optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-4}),
                  ('adadelta', optim.Adadelta, {}),
                  ('rmsprop', optim.RMSprop, {'eps': 1e-5}),
                  ('adam', optim.Adam, {'eps': 1e-5})]

optim_name, optimizer, opt_kargs = optimizer_list[optimizer_set]

seed_list = [42, 25, 84, 12, 90]
seed = seed_list[seed_set]
torch.manual_seed(seed)


cifar10_trainset =  dataset('data/', train=True, transform=train_transform, download=True)
cifar10_testset =  dataset('data/', train=False, transform=test_transform, download=True)

total_size = len(cifar10_trainset)
split = int(total_size * mi_split)

# indices = list(range(total_size))
# target_train_idx = indices[:-split]
# target_out_idx = indices[-split:]

# target_train_sampler = SubsetRandomSampler(target_train_idx)
# target_out_sampler = SubsetRandomSampler(target_out_idx)

# target_train_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, sampler=target_train_sampler)
# target_out_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, sampler=target_out_sampler)


target_train_loader = torch.utils.data.DataLoader(cifar10_trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=num_workers)

target_out_loader = torch.utils.data.DataLoader(cifar10_testset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=num_workers)



target_net = target_net_type()
try:
    target_net = target_net_type().to(device)
except RuntimeError as e:
    logger.warning(f'RuntimeError ({e}) was raised when trying to move target net to device. Skipping that step')
# target_net.apply(weights_init) # ?!? THE DEVIL

target_loss = nn.CrossEntropyLoss()
optimizer = optimizer(target_net.parameters(), lr=lr, **opt_kargs) # optim.SGD(target_net.parameters(), lr=lr, momentum=0.9)
target_optim = {
    'optimizer': optimizer,
    }
    
if train_scheduler: # TODO For now it just means learning rate decay
    target_optim['train_scheduler'] = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    optim_name = f'{optim_name}-adaptativeLR'
    

# attack_net = mlleaks_mlp(n_in=k)
# try:
#     attack_net.to(device)
 # except RuntimeError as e:
#     logger.warning(f'RuntimeError ({e}) was raised when trying to move attack net to device. Skipping that step')
# attack_net.apply(weights_init)

# attack_loss = nn.BCELoss()
# attack_optim = optim.SGD(attack_net.parameters(), lr=lr, momentum=0.9)


from src.membership_inference import mi_wrapper as mi
logger.info('Start training')
mi.train_with_mi(model=target_net,
                train_dataloader=target_train_loader,
                criterion=target_loss,
                optimizer=target_optim,
                device=device,
                out_dataloader=target_out_loader,
                attack_model=None,
                attack_criterion=None,
                attack_optimizer=None,
                attack_batch_size = 128,
                save_data = False, #'MI_data/saved_data',
                mi_epochs='all',
                epochs=n_epochs,
                attack_epochs=attack_epochs,
                log_iteration = 1, start_epoch=0,
                logger_name=f'logsAdam_saveme/{dataset_to_use}/{model_to_use}/{optim_name}-{lr}',
                force_new_model=clean_start)
