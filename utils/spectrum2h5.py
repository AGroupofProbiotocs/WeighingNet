"""
Created on Sun Feb 16 14:45:21 2020

@author: Dragon
"""

import h5py
from utils import spectrogram_double_channel
import os

dataset = 'val'
data_len = 6000
freq = 2000
sub_freq = 2000
pad_to = 150
seg_len = 51
noverlap = 12


DATA_FOLDER = '/home/shared_folders/Radar_Gesture_Data/' + dataset
H5_FOLDER = '../data/' + dataset

gesture_folder = [os.path.join(DATA_FOLDER, x) for x in os.listdir(DATA_FOLDER)]
# print(gesture_folder)

for cur_folder in gesture_folder:
    a = os.listdir(cur_folder)
    radar_data_path = os.path.join(cur_folder, os.listdir(cur_folder)[0])
#    print(radar_data_path)
    
    radar_spec = spectrogram_double_channel(radar_data_path, data_len=data_len, seg_len = seg_len,
                                            noverlap = noverlap, freq = freq, sub_freq=sub_freq,
                                            pad_to = pad_to, shuffle=False, shift=True,
                                            log=True, normalization=False)
    # print(radar_spec.shape)

    gesture_name = cur_folder.split('/')[-1]
    h5_path = os.path.join(H5_FOLDER, gesture_name)
    if not os.path.exists(h5_path):
        os.mkdir(h5_path)

    h5_file = os.path.join(h5_path, gesture_name + '.h5')

    with h5py.File(h5_file,'w') as f:
        f['data'] = radar_spec[..., None]  #(?, 150, 153, 1)

    print("folder %s success"%gesture_name)

