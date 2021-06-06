
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import h5py

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def collect_gesture_folders(data_folder = r'gesture_spec_data/trainval', num_train=70):

    gesture_folders = [os.path.join(data_folder, gesture_name) \
                       for gesture_name in os.listdir(data_folder)]
    random.seed(1)
    random.shuffle(gesture_folders)
    # print(len(gesture_folders))

    metatrain_gesture_folders = gesture_folders[:num_train]
    metaval_gesture_folders = gesture_folders[num_train:]

    return metatrain_gesture_folders, metaval_gesture_folders


def generate_meta_data(gesture_folders, num_classes, support_num, query_num, shuffle=True, normalize=False, standardize=False):
    '''

    :param gesture_folders: a list containing data path of different characters
    :param num_classes: 5 in default for training
    :param support_num: 1 for support
    :param query_num: 19 for query
    '''

    class_folders = np.random.choice(gesture_folders, num_classes, replace=False) #sample 5 characters

    support_data = [] #store all(1x5) path of the training images
    query_data = [] #store all(19x5) path of the test images
    # support_labels = []
    query_labels = []

    for i, folder in enumerate(class_folders):
        cur_path = os.path.join(folder, folder.split('/')[-1]+'.h5')
        with h5py.File(cur_path, 'r') as f:
            cur_data = np.array(f.get('data'))
            np.random.shuffle(cur_data)

        support_data.append(cur_data[:support_num]) #1 image for each character/class
        query_data.append(cur_data[support_num:support_num+query_num])  #19 image for each character/class

        # self.support_labels += [i]*support_num
        query_labels += [i]*query_num

    support_data = np.concatenate(support_data)
    query_data = np.concatenate(query_data)
    # self.support_labels = np.array(self.support_labels)
    query_labels = np.array(query_labels, dtype='int32')

    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(query_data)
        np.random.set_state(state)
        np.random.shuffle(query_labels)

    if normalize:
        support_data = support_data/1000
        query_data = query_data/1000

    return support_data, query_data, query_labels








