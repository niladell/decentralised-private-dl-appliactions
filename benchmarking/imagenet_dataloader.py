import torch
from torchvision import transforms
import os
import h5py
import tqdm
import logging
import csv
import numpy as np
logger = logging.getLogger(__name__)




class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform='Default', data_registry='saved_registry.csv'):
        # super.__init__()
        self.classes = sorted([f'{folder}/{x}' for x in os.listdir(folder) if 'imagenet_n' in x])
        logger.info(f'Imagenet class datafiles: {len(self.classes)}')
        self.labels = []
        # self.class_els = {}
        self.data_pointer = []
        self.data_files = {}
        label = 0
        if any(data_registry in s for s in os.listdir(folder)):
            with open(f'{folder}/{data_registry}', 'r') as f:
                self.data_pointer = list(csv.reader(f))
            # for clss in tqdm.tqdm(self.classes):
            #     h5f = h5py.File(clss, 'r')
            #     self.data_files[clss] = h5f
        else:
            logger.warning('Class registry not found -- a new one will be created. This is a very slow process')
            for clss in tqdm.tqdm(self.classes):
                h5f = h5py.File(clss, 'r') # TODO why the fk is this so slow with %timeit I get ~1ms on this op
                pics = list(h5f.keys())
                pics.sort(key=lambda x: int(x))
                # self.class_els[clss] = pics
                self.data_pointer += [(clss, key, label) for key in pics]
                label += 1
                # self.data_files[clss] = h5f
            if data_registry:
                with open(f'{folder}/{data_registry}', 'w') as f:
                    wr = csv.writer(f)
                    wr.writerows(self.data_pointer)
                logger.info('Finished writing regsitry file')

        self.transform = transform
        if transform == 'Default':
            # Taken from https://github.com/pytorch/vision/issues/39#issuecomment-403701432
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

            self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                # transforms.RandomSizedCrop(224),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])

    def __len__(self):
        return len(self.data_pointer)

    def __getitem__(self, idx):
        h5_file, key, label = self.data_pointer[idx]

        with h5py.File(h5_file, 'r') as h5f:
            image = h5f[key][:]
        # image = image / 255
        # image = np.swapaxes(image, -1, 0)
        # logger.info(image)
        # logger.info(self.data_pointer[idx])

        # image = self.data_files[h5_file][key][:]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(label)) # idk if it'd be better to first check for the type

        return image, label

    def close(self):
        for f in self.data_files.values():
            f.close
