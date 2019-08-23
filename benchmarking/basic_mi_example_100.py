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

import logging
logFormatter ='%(asctime)s %(levelname)s -- %(name)s {%(module)s} [%(funcName)s]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logFormatter, datefmt='%m-%d,%H:%M:%S')

logger = logging.getLogger(__name__)

logger.info("Python: %s" % sys.version)
logger.info("Pytorch: %s" % torch.__version__)


import dpsgd

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

from model_dataset_loader import get_dataset_and_model

model, loaders = get_dataset_and_model(dataset_to_use, model_to_use, batch_size, data_aug)
if data_aug:
    dataset_to_use = f'{dataset_to_use}-Aug-'

# ('sgd', optim.SGD), , ('adadelta', optim.Adadelta, {}),
# optimizer_list = [  ('rmsprop', optim.RMSprop, {'eps': 1e-5})]
# optimizer_list = [('adam', optim.Adam, {'eps': 1e-5})]
# optimizer_list = [('sgd', optim.SGD, {})]


optimizer_list = [('sgd', optim.SGD, {}),
                  ('sgd_momentum', optim.SGD, {'momentum': 0.9}),
                  ('sgd_momentum_wd', optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-4}),
                  ('sgd_momentum_wd-5e-3', optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-3}),
                  ('adadelta', optim.Adadelta, {}),
                  ('rmsprop', optim.RMSprop, {'eps': 1e-5}),
                  ('adam', optim.Adam, {'eps': 1e-5}),
                  ('dp_sgd', dpsgd.DPSGD, {'batch_size': batch_size, 'C': 1, 'noise_multiplier': 1})]

optim_name, optimizer, opt_kargs = optimizer_list[optimizer_set]

seed_list = [42, 25, 84, 12, 90]
seed = seed_list[seed_set]
torch.manual_seed(seed)


target_net = model()
try:
    target_net = model().to(device)
except RuntimeError as e:
    logger.warning(f'RuntimeError ({e}) was raised when trying to move target net to device. Skipping that step')
# target_net.apply(weights_init) # ?!? THE DEVIL

target_loss = nn.CrossEntropyLoss()
optimizer = optimizer(target_net.parameters(), lr=lr, **opt_kargs) # optim.SGD(target_net.parameters(), lr=lr, momentum=0.9)
target_optim = {
    'optimizer': optimizer,
    }
    
if train_scheduler: # TODO For now it just means learning rate decay
    # target_optim['train_scheduler'] = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    # optim_name = f'{optim_name}-adaptativeLR'
    target_optim['train_scheduler'] = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 85, 95], gamma=0.2) #learning rate decay
    optim_name = f'{optim_name}-adaptativeLR-60.85.95'

target_train_loader = loaders['trainloader']
target_out_loader = loaders['testloader']
mi_trainset_loader = loaders['mi_trainloader']


from src.membership_inference import mi_wrapper as mi
logger.info('Start training')
mi.train_with_mi(model=target_net,
                train_dataloader=target_train_loader,
                criterion=target_loss,
                optimizer=target_optim,
                device=device,
                out_dataloader=target_out_loader,
                # attack_model=None,
                attack_criterion=None,
                attack_optimizer=None,
                attack_batch_size = 128,
                save_data = 'test_Aug14', #False, #'MI_data/saved_data',
                mi_epochs='all',
                epochs=n_epochs,
                log_iteration = 1, start_epoch=0,
                logger_name=f'logs_test0819/{dataset_to_use}-MI/{model_to_use}/{optim_name}-{lr}.seed{seed}',
                force_new_model=clean_start,
                mi_train_loader=mi_trainset_loader)
