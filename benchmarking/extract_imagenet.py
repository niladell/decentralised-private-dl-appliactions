import tarfile
import os
import shutil
import h5py
import cv2
import tqdm
import sys
base_dir = 'data/train'
files = os.listdir(base_dir)
try:
    shutil.rmtree(f'{base_dir}/tmp')
except FileNotFoundError as e:
    print(e)
with h5py.File('imagenet.h5', 'w') as h5f:
    label_arr = []
    record_names = []
    label = 0
    count = 0
    for i_f, f in tqdm.tqdm(enumerate(files)): 
        if 'tar' in f:
            os.mkdir(f'{base_dir}/tmp')
            file_name = f.split('.')[0]
            tar = tarfile.open(f'{base_dir}/{f}')
            tar.extractall(path=f'{base_dir}/tmp')
            images = os.listdir(f'{base_dir}/tmp')
            record_names.append(file_name)

            for i, image in enumerate(images):
                im = cv2.imread(f'{base_dir}/tmp/{image}')
                # tqdm.tqdm.write(f'{im.shape}')
                h5f.create_dataset(f'{count}', data=im)

                label_arr.append(label)
                count += 1
            print(f'{i_f}/{len(files)} -- {len(list(h5f.keys()))}'  )
            sys.stdout.flush()

            tar.close()
            shutil.rmtree(f'{base_dir}/tmp') 
            label += 1