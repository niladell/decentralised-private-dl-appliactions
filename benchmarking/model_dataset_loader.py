import numpy as np
import torch
import torchvision
from utils_basic_mi_example import AlexNet, mlleaks_mlp, weights_init

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

def get_dataset_and_model(dataset, model, batch_size, data_aug, num_workers=8):
    assert dataset in dataset_list.keys(), f'Ilegal Dataset {dataset}'

    if dataset != 'CIFAR100':
        raise NotImplementedError

    model_to_use = model  # TODO Refactor
    dataset_to_use = dataset
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
    
    ## GET DATASET
    dataset, num_classes = dataset_list[dataset_to_use]
    train_transform, test_transform, mi_transform = get_transforms(dataset_to_use, data_aug)

    trainset =  dataset('data/', train=True, transform=train_transform, download=True)
    testset =  dataset('data/', train=False, transform=test_transform, download=True)
    # In the case of data augmentation trainset and MI_trainset will be different. Is that rational¿
    # at the end of the day it does really depend on how you define MI, it's not the pure definition of "samples
    # used during on training" if you take into account that you are modifying them (i.e. practically changing de
    # respective distributions of train and tests MI sets)
    mi_trainset =  dataset('data/', train=True, transform=mi_transform, download=True)
    
    target_train_loader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                #   drop_last=True,
                                                  num_workers=num_workers)

    target_test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    # drop_last=True,
                                                    num_workers=num_workers)
    
    mi_train_sampler = torch.utils.data.SubsetRandomSampler(
                            np.random.randint(len(trainset), size=len(testset))
                            )
    mi_train_loader = torch.utils.data.DataLoader(mi_trainset,
                                                    batch_size=batch_size,
                                                    sampler=mi_train_sampler,
                                                    # shuffle=True,
                                                    # drop_last=True,
                                                    num_workers=num_workers)
    
    loaders = {
        'trainloader': target_train_loader,
        'testloader': target_test_loader,
        'mi_trainloader': mi_train_loader}

    return target_net_type, loaders 

def get_transforms(dataset_to_use: str, data_aug: bool):
    if dataset_to_use == 'CIFAR100':
        return _CIFAR100_transform(data_aug)
    else:
        raise NotImplementedError(f'No transform for {dataset_to_use}')


def _CIFAR100_transform(data_aug: bool):
    if data_aug:
        train_transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
        # dataset_to_use = f'{dataset_to_use}-wAug'
        mi_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])

    else:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
        mi_transform = train_transform

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])

    return train_transform, test_transform, mi_transform