import numpy as np
import h5py
import os

dataset = ['train', 'test', 'val']
all_data = []

for ds in dataset:
    H5_FOLDER = './gesture_spec_data/raw_' + ds
    gesture_folders = [os.path.join(H5_FOLDER, gesture_name) for gesture_name in os.listdir(H5_FOLDER)]
    for fd in gesture_folders:
        cur_path = os.path.join(fd, fd.split('/')[-1] + '.h5')
        with h5py.File(cur_path, 'r') as f:
            cur_data = np.array(f.get('data'))
        all_data.append(cur_data)
        print(fd.split('/')[-1], 'finished!')

all_data = np.concatenate(all_data)/1000
print('concatenation finished!')
print('\nmax:', np.max(all_data), '\nmin:', np.min(all_data))
mean = np.mean(all_data)
std = np.std(all_data)
print('\nmean:', mean, '\nstd:', std)
