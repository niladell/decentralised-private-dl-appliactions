import tarfile
import os
import shutil
import h5py
import cv2
import tqdm
import sys

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--chunk')

args = parser.parse_args()


datafile = args.chunk
datafile = datafile.split('/')[-1]

base_dir = 'data/train'
files = os.listdir(base_dir)
# try:
#     shutil.rmtree(f'{base_dir}/tmp')
# except FileNotFoundError as e:
#     print(e)
if 'tar' in datafile:
    file_name = datafile.split('.')[0]
    with h5py.File(f'imagenet_{file_name}.h5', 'w') as h5f:
        # label_arr = []
        # record_names = []
        label = 0
        # count = 0
        # for i_f, f in tqdm.tqdm(enumerate(files)): 
        f = datafile
        try:
            os.mkdir(f'{base_dir}/{file_name}')
        except:
            pass
        tar = tarfile.open(f'{base_dir}/{f}')
        tar.extractall(path=f'{base_dir}/{file_name}')
        images = os.listdir(f'{base_dir}/{file_name}')
        # record_names.append(file_name)

        for i, image in enumerate(images):
            try:
                im = cv2.imread(f'{base_dir}/{file_name}/{image}')
                # tqdm.tqdm.write(f'{im.shape}')
                h5f.create_dataset(f'{i}', data=im)
            except TypeError as e:
                print(f'{e} with {file_name} on {image}: {im}')
            # label_arr.append(label)
            # count += 1

        # print(f'{i_f}/{len(files)} -- {len(list(h5f.keys()))}'  )
        sys.stdout.flush()

        tar.close()
        shutil.rmtree(f'{base_dir}/{file_name}') 
        label += 1