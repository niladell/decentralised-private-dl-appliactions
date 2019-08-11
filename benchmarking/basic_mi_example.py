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
logFormatter ='%(asctime)s:%(levelname)s:%(name)s [%(funcName)s] -- %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logFormatter, datefmt='%m%d,%H:%M:%S')

logger = logging.getLogger(__name__)

logger.info("Python: %s" % sys.version)
logger.info("Pytorch: %s" % torch.__version__)

# determine device to run network on (runs on gpu if available)
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_epochs = 100
attack_epochs = 10
batch_size = 128
k = 10
mi_split = 0.2 # CIFAR
# lr = 0.001
num_workers = 16

# dataset_to_use = 'CIFAR10'
# model_to_use = 'vgg16'

dropout = 'default'

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--optimizer_set', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--clean_start', type=str)
args = parser.parse_args()
logger.info(args)


optimizer_set = args.optimizer_set
seed_set = args.seed
model_to_use = args.model
dataset_to_use = args.dataset
lr = args.lr
clean_start = args.clean_start.lower() == 'true' 

if dataset_to_use == 'ImageNet': #TODO Properly organize these different options¿ YML file?
    mi_split = 0.05

if model_to_use == 'vanilla':
    target_net_type = models.mlleaks_cnn
    shadow_net_type = models.mlleaks_cnn
elif model_to_use == 'alexnet':
    if dataset_to_use == 'ImageNet':
        target_net_type = torchvision.models.alexnet
        shadow_net_type = torchvision.models.alexnet
    else:
        target_net_type = AlexNet #lambda: torchvision.models.alexnet(num_classes=10)
        shadow_net_type = AlexNet #lambda: torchvision.models.alexnet(num_classes=10)
elif model_to_use == 'vgg16':
    if dataset_to_use == 'ImageNet':
        batch_size = 64
        target_net_type = torchvision.models.vgg16
        shadow_net_type = torchvision.models.vgg16
    else: 
        import vgg
        target_net_type = lambda: vgg.vgg16_bn()
        shadow_net_type = lambda: vgg.vgg16_bn() 
elif model_to_use == 'resnet':
    if dataset_to_use == 'ImageNet':
            batch_size = 64    # import resnet
            target_net_type = torchvision.models.resnet50
            shadow_net_type = torchvision.models.resnet50
    else:
        import resnet
        target_net_type = lambda: resnet.resnet32(num_classes=10)
        target_net_type = lambda: resnet.resnet32(num_classes=10)
        resnet.test(target_net_type())
elif model_to_use == 'inception_v3':
    target_net_type = torchvision.models.inception_v3
    shadow_net_type = torchvision.models.inception_v3
elif model_to_use == 'squeezenet':
    target_net_type = torchvision.models.squeezenet1_1
    shadow_net_type = torchvision.models.squeezenet1_1
else:
    raise(NotImplementedError)


# def get_Imagenet(*args, **kwargs):
#     if kwargs['train']:
#         kwargs['split'] = 'train'
#     else:
#         kwargs['split'] = 'val'
#     del kwargs['train']
#     return torchvision.datasets.ImageNet(*args, **kwargs)

def get_Imagenet(*args, **kwargs):
    folder = 'data/Imagenet/train'
    from imagenet_dataloader import ImagenetDataset
    return ImagenetDataset(folder)


def get_SVHN(*args, **kwargs):
    if kwargs['train']:
        kwargs['split'] = 'train'
    else:
        kwargs['split'] = 'test'
    del kwargs['train']
    return torchvision.datasets.SVHN(*args, **kwargs)


dataset_list = {
    'CIFAR10': torchvision.datasets.CIFAR10,
    'CIFAR100':  torchvision.datasets.CIFAR100,
    'SVHN': get_SVHN,
    'ImageNet': get_Imagenet
}

dataset = dataset_list[dataset_to_use]

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# ('sgd', optim.SGD), , ('adadelta', optim.Adadelta, {}), 
# optimizer_list = [  ('rmsprop', optim.RMSprop, {'eps': 1e-5})]
# optimizer_list = [('adam', optim.Adam, {'eps': 1e-5})]
# optimizer_list = [('sgd', optim.SGD, {})]


optimizer_list = [('sgd', optim.SGD, {}),
                  ('sgd_momentum', optim.SGD, {'momentum': 0.9}),
                  ('adadelta', optim.Adadelta, {}),
                  ('rmsprop', optim.RMSprop, {'eps': 1e-5}),
                  ('adam', optim.Adam, {'eps': 1e-5})]

optim_name, optimizer, opt_kargs = optimizer_list[optimizer_set]

seed_list = [42, 25, 84, 12, 90]
seed = seed_list[seed_set]

# for optim_name, optimizer, opt_kargs in optimizer_list:
#     for seed in seed_list:
#         torch.manual_seed(seed)


cifar10_trainset =  dataset('data/', train=True, transform=train_transform, download=True)
# cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

total_size = len(cifar10_trainset)
split = int(total_size * mi_split)

# indices = list(range(total_size))
np.random.seed(seed)
indices = np.random.permutation(total_size)
target_train_idx = indices[:-split]
target_out_idx = indices[-split:]

target_train_sampler = SubsetRandomSampler(target_train_idx)
target_out_sampler = SubsetRandomSampler(target_out_idx)

target_train_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, sampler=target_train_sampler,num_workers =num_workers)
target_out_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size//2, sampler=target_out_sampler, num_workers =num_workers)

target_net = target_net_type()
try:
    target_net = target_net_type().to(device)
except RuntimeError as e:
    logger.warning(f'RuntimeError ({e}) was raised when trying to move target net to device. Skipping that step')
# target_net.apply(weights_init)

target_loss = nn.CrossEntropyLoss()
target_optim = optimizer(target_net.parameters(), lr=lr, **opt_kargs) # optim.SGD(target_net.parameters(), lr=lr, momentum=0.9)

attack_net = mlleaks_mlp(n_in=k)
try:
    attack_net.to(device)
except RuntimeError as e:
    logger.warning(f'RuntimeError ({e}) was raised when trying to move attack net to device. Skipping that step')    
attack_net.apply(weights_init)

attack_loss = nn.BCELoss()
attack_optim = optim.SGD(attack_net.parameters(), lr=lr, momentum=0.9)


from src.membership_inference import mi_wrapper as mi
logger.info('Start training')
mi.train_with_mi(model=target_net,
                train_dataloader=target_train_loader,
                criterion=target_loss,
                optimizer=target_optim,
                device=device,
                attack_model=attack_net,
                out_dataloader=target_out_loader,
                attack_criterion=attack_loss,
                attack_optimizer=attack_optim,
                attack_batch_size = batch_size,
                save_data = False, #'MI_data/{datdataset_to_useaset_}_{model_to_use}/saved_data_',
                mi_epochs='all',
                epochs=n_epochs,
                attack_epochs=attack_epochs,
                log_iteration = 50, start_epoch=0, 
                logger_name=f'logs/{dataset_to_use}/{model_to_use}-{dropout}/{optim_name}-{lr}',
                force_new_model=clean_start)