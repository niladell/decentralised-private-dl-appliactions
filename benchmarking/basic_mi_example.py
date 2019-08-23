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

# clean_start = args.clean_start.lower() == 'true' 

# if dataset_to_use == 'ImageNet': #TODO Properly organize these different options¿ YML file?
#     mi_split = 0.05

# if model_to_use == 'vanilla':
#     target_net_type = models.mlleaks_cnn
#     shadow_net_type = models.mlleaks_cnn
# elif model_to_use == 'alexnet':
#     if dataset_to_use == 'ImageNet':
#         target_net_type = torchvision.models.alexnet
#         shadow_net_type = torchvision.models.alexnet
#     else:
#         target_net_type = AlexNet #lambda: torchvision.models.alexnet(num_classes=10)
#         shadow_net_type = AlexNet #lambda: torchvision.models.alexnet(num_classes=10)
# elif model_to_use == 'vgg16':
#     if dataset_to_use == 'ImageNet':
#         batch_size = 64
#         target_net_type = torchvision.models.vgg16
#         shadow_net_type = torchvision.models.vgg16
#     else: 
#         import vgg
#         target_net_type = lambda: vgg.vgg16_bn()
#         shadow_net_type = lambda: vgg.vgg16_bn() 
# elif model_to_use == 'resnet-50':
#     if dataset_to_use == 'ImageNet':
#             batch_size = 64    # import resnet
#             target_net_type = torchvision.models.resnet50
#             shadow_net_type = torchvision.models.resnet50
# elif model_to_use == 'resnet-32':
#     import resnet
#     target_net_type = lambda: resnet.resnet32(num_classes=10)
#     target_net_type = lambda: resnet.resnet32(num_classes=10)
#     resnet.test(target_net_type())
# elif model_to_use == 'inception_v3':
#     target_net_type = torchvision.models.inception_v3
#     shadow_net_type = torchvision.models.inception_v3
# elif model_to_use == 'squeezenet':
#     target_net_type = torchvision.models.squeezenet1_1
#     shadow_net_type = torchvision.models.squeezenet1_1
# else:
#     raise(NotImplementedError)

# def get_Imagenet(*args, **kwargs):
#     folder = 'data/Imagenet/train'
#     from imagenet_dataloader import ImagenetDataset
#     return ImagenetDataset(folder)


# def get_SVHN(*args, **kwargs):
#     if kwargs['train']:
#         kwargs['split'] = 'train'
#     else:
#         kwargs['split'] = 'test'
#     del kwargs['train']
#     return torchvision.datasets.SVHN(*args, **kwargs)


# dataset_list = {
#     'CIFAR10': torchvision.datasets.CIFAR10,
#     'CIFAR100':  torchvision.datasets.CIFAR100,
#     'SVHN': get_SVHN,
#     'ImageNet': get_Imagenet
# }

# dataset = dataset_list[dataset_to_use]

from model_dataset_loader import get_dataset_and_model

model, loaders = get_dataset_and_model(dataset_to_use, model_to_use, batch_size, data_aug)
if data_aug:
    dataset_to_use = f'{dataset_to_use}-Aug-'

# optimizer_list = [  ('rmsprop', optim.RMSprop, {'eps': 1e-5})]
# optimizer_list = [('adam', optim.Adam, {'eps': 1e-5})]
# optimizer_list = [('sgd', optim.SGD, {})]


optimizer_list = [('sgd', optim.SGD, {}),
                  ('sgd_momentum', optim.SGD, {'momentum': 0.9}),
                  ('sgd_momentum_wd', optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-4}),
                  ('sgd_momentum_wd-5e-3', optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-3}),
                  ('adadelta', optim.Adadelta, {}),
                  ('rmsprop', optim.RMSprop, {'eps': 1e-5}),
                  ('adam', optim.Adam, {'eps': 1e-5})]

optim_name, optimizer, opt_kargs = optimizer_list[optimizer_set]

seed_list = [42, 25, 84, 12, 90]
seed = seed_list[seed_set]
torch.manual_seed(seed)

target_train_loader = loaders['trainloader']
target_out_loader = loaders['testloader']
mi_trainset_loader = loaders['mi_trainloader']



try:
    target_net = model().to(device)
except RuntimeError as e:
    logger.warning(f'RuntimeError ({e}) was raised when trying to '
                    'move target net to device. Skipping that step')
    target_net = model()

target_loss = nn.CrossEntropyLoss()
target_optim = optimizer(target_net.parameters(), lr=lr, **opt_kargs) # optim.SGD(target_net.parameters(), lr=lr, momentum=0.9)

if dataset_to_use == 'CIFAR10':
    topk=(1,)

from src.membership_inference import mi_wrapper as mi
logger.info('Start training')
mi.train_with_mi(model=target_net,
                train_dataloader=target_train_loader,
                criterion=target_loss,
                optimizer=target_optim,
                device=device,
                out_dataloader=target_out_loader,
                attack_criterion=None,
                attack_optimizer=None,
                attack_batch_size = batch_size,
                save_data = False, #'MI_data/{datdataset_to_useaset_}_{model_to_use}/saved_data_',
                mi_epochs='all',
                epochs=n_epochs,
                log_iteration = 50, start_epoch=0, 
                logger_name=f'logs_new_0822/{dataset_to_use}/{model_to_use}-{dropout}/{optim_name}-{lr}',
                force_new_model=clean_start,
                mi_train_loader=mi_trainset_loader,
                topk=topk)
