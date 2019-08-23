import numpy as np
import torch
from torchvision import transforms, datasets, models
from utils_basic_mi_example import AlexNet, mlleaks_mlp, weights_init

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
    return datasets.SVHN(*args, **kwargs)


dataset_list = {
        'CIFAR10': (datasets.CIFAR10, 10),
        'CIFAR100':  (datasets.CIFAR100, 100),
        'SVHN': (get_SVHN, 10),
        'ImageNet': (get_Imagenet, 1000)
    }

def get_dataset_and_model(dataset, model, batch_size, data_aug, num_workers=8):
    model_to_use = model  # TODO Refactor
    dataset_to_use = dataset
    assert dataset_to_use in dataset_list.keys(), f'Ilegal Dataset {dataset_to_use}'

    dataset, num_classes = dataset_list[dataset_to_use]
    if model_to_use == 'vanilla':
        target_net_type = models.mlleaks_cnn
    elif model_to_use == 'alexnet':
        target_net_type = lambda: models.alexnet(num_classes=num_classes)
    elif model_to_use == 'vgg16':
        if num_classes == 100:
            import vgg_100
            target_net_type = lambda: vgg_100.vgg16_bn()
        elif num_classes == 10:
            import vgg
            target_net_type = lambda: vgg.vgg16_bn()
    elif model_to_use == 'resnet-32':
        import resnet
        target_net_type = lambda: resnet.resnet32(num_classes=num_classes)
        resnet.test(target_net_type())
    elif model_to_use == 'resnet-34':
        import resnet_new as res
        target_net_type = lambda: res.resnet34(num_classes=num_classes)
    elif model_to_use == 'resnet-50':
        if dataset_to_use == 'ImageNet':
                batch_size = 32    # import resnet
                target_net_type = models.resnet50
        else:
            target_net_type = res.resnet50(num_classes=num_classes)
    elif model_to_use == 'inception_v3':
        target_net_type = models.inception_v3
    else:
        raise(NotImplementedError)
    
    ## GET DATASET
    train_transform, test_transform, mi_transform = get_transforms(dataset_to_use, data_aug)

    trainset =  dataset('data/', train=True, transform=train_transform, download=True)
    testset =  dataset('data/', train=False, transform=test_transform, download=True)
    # In the case of data augmentation trainset and MI_trainset will be different. Is that rational¿
    # at the end of the day it does really depend on how you define MI, it's not the pure definition of "samples
    # used during on training" if you take into account that you are modifying them (i.e. practically changing de
    # respective distributions of train and tests MI sets)
    mi_trainset =  dataset('data/', train=True, transform=mi_transform, download=True)
    
    if dataset_to_use == 'ImageNet':
        split = 0.9
        idxs = np.random.permutation(len(trainset))
        cut = int(split * len(idxs))
        train_idxs, test_idxs = idxs[:cut], idxs[cut:]
        mi_idxs = np.random.choice(train_idxs, size=(len(test_idxs)))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idxs)
        mi_train_sampler = torch.utils.data.SubsetRandomSampler(mi_idxs) 

        target_train_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    #   drop_last=True,
                                                    num_workers=num_workers)

        target_test_loader = torch.utils.data.DataLoader(testset,
                                                        batch_size=batch_size,
                                                        sampler=test_sampler,
                                                        # drop_last=True,
                                                        num_workers=num_workers)
        
        mi_train_loader = torch.utils.data.DataLoader(mi_trainset,
                                                        batch_size=batch_size,
                                                        sampler=mi_train_sampler,
                                                        # shuffle=True,
                                                        # drop_last=True,
                                                        num_workers=num_workers)
        loaders = {
            'trainloader': target_train_loader,
            'testloader': target_test_loader,
            'mi_trainloader': mi_train_loader
        }
        return target_net_type, loaders 

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
    elif dataset_to_use == 'CIFAR10':
        return _CIFAR10_transform(data_aug)
    elif dataset_to_use == 'ImageNet':
        return _Imagenet_transform(data_aug)
    else:
        raise NotImplementedError(f'No transform for {dataset_to_use}')


def _Imagenet_transform(data_aug: bool):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # SUbset sampler for test/MI w/o data aug
    test_transform = mi_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if data_aug:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomSizedCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform, mi_transform


def _CIFAR10_transform(data_aug: bool):
    if data_aug:
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
        ])
        # dataset_to_use = f'{dataset_to_use}-wAug'
        mi_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
        ])

    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
        ])
        mi_transform = train_transform

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
    ])

    return train_transform, test_transform, mi_transform

def _CIFAR100_transform(data_aug: bool):
    if data_aug:
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
        # dataset_to_use = f'{dataset_to_use}-wAug'
        mi_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])

    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
        mi_transform = train_transform

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])

    return train_transform, test_transform, mi_transform