# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import h5py
from utils.utils import spectrogram_double_channel

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def collect_gesture_folders(data_folder = r'gesture_spec_data/trainval', num_train=70):

    gesture_folders = [os.path.join(data_folder, gesture_name) \
                       for gesture_name in os.listdir(data_folder)]
    random.seed(1)
    random.shuffle(gesture_folders)
    # print(len(gesture_folders))

    metatrain_gesture_folders = gesture_folders[:num_train]
    metaval_gesture_folders = gesture_folders[num_train:]

    return metatrain_gesture_folders, metaval_gesture_folders

class GestureTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_gesture_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, gesture_folders, num_classes, support_num, query_num, shuffle_all=True, standardize=False,
                 log = False, threshold = None):
        '''

        :param gesture_folders: a list containing data path of different characters
        :param num_classes: 5 in default for training
        :param support_num: 1 for support
        :param query_num: 19 for query
        '''
        self.shuffle_all = shuffle_all
        self.standardize = standardize
        self.gesture_folders = gesture_folders
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num
        self.log = log
        self.threshold = threshold

        class_folders = np.random.choice(self.gesture_folders, self.num_classes, replace=False) #sample 5 characters

        self.support_data = [] #store all(1x5) path of the training images
        self.query_data = [] #store all(19x5) path of the test images
        self.support_labels = []
        self.query_labels = []

        for i, folder in enumerate(class_folders):
            cur_path = os.path.join(folder, folder.split('/')[-1]+'.h5')
            with h5py.File(cur_path, 'r') as f:
                cur_data = np.array(f.get('data'))
            if self.shuffle_all:
                random.shuffle(cur_data)
                self.support_data.append(cur_data[:support_num])  # 1 image for each character/class
                self.query_data.append(cur_data[support_num:support_num + query_num])  # 19 image for each character/class
            else:
                person_list = [cur_data[i*5:(i+1)*5] for i in range(4)]
                random.shuffle(person_list)
                cur_data = np.concatenate(person_list)
                cur_data_2 = cur_data[support_num:support_num + query_num]
                random.shuffle(cur_data_2)
                self.support_data.append(cur_data[:support_num])
                #for observing the difference between samples from same volunteer and different volunteers
                # a = random.randint(support_num, 4)
                # b = random.randint(5, 19)
                self.query_data.append(cur_data_2)

            self.support_labels += [i]*support_num
            self.query_labels += [i]*query_num

        self.support_data = np.concatenate(self.support_data)
        self.query_data = np.concatenate(self.query_data)


class GestureDataset(Dataset):

    def __init__(self, task, split='support'):
        self.task = task
        self.split = split
        self.image_data = self.task.support_data if self.split == 'support' else self.task.query_data
        self.labels = self.task.support_labels if self.split == 'support' else self.task.query_labels
        self.transform = transforms.Normalize(mean=[0.01304], std=[0.06052])

    def __getitem__(self, idx):
        image = self.image_data[idx]
        if self.task.log:
            image = 10 * np.log10(image + np.spacing(1))
            if self.task.threshold is not None:
                image[image < self.task.threshold] = np.min(image)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        if self.task.standardize:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.labels)

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='support',shuffle=True):
    # NOTE: batch size here is # instances PER CLASS

    dataset = GestureDataset(task, split=split)

    if split == 'support':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

def get_specified_data(gesture_folders, num_classes, support_num):
    ORG_DATA_FOLDER = '/home/shared_folders/Radar_Gesture_Data/test'
    class_folders = np.random.choice(gesture_folders, num_classes, replace=False)  # sample 5 characters

    support_data = []  # store all(1x5) path of the training images
    query_data = []  # store all(19x5) path of the test images
    support_labels = []
    query_labels = []
    support_t = []
    support_freq = []
    query_t = []
    query_freq = []

    for i, folder in enumerate(class_folders):
        cur_path = ORG_DATA_FOLDER + '/' + folder.split('/')[-1] + '/' + folder.split('/')[-1] + '.h5'
        # print(cur_path)
        cur_data, coors = spectrogram_double_channel(cur_path, data_len=6000, seg_len=51,
                                                    noverlap=12, freq=2000, sub_freq=2000,
                                                    pad_to=150, shuffle=False, shift=True,
                                                    log=True, normalization=False, need_coor=True,
                                                    threshold=4)
        support_idx = random.randint(0, 19)
        support_data.append(cur_data[support_idx:support_idx+1])  # 1 image for each character/class
        support_t.append(coors[0][support_idx:support_idx+1])
        support_freq.append(coors[1][support_idx:support_idx+1])
        rem = support_idx % 5
        div = support_idx // 5
        a = list(range(5))
        b = list(range(4))
        a.remove(rem)
        b.remove(div)
        query_idx_1 = div*5 + random.choice(a)
        query_idx_2 = random.choice(b)*5 + random.randint(0,4)
        print("support_idx: ", support_idx)
        print("query_idx_1: ", query_idx_1)
        print("query_idx_2: ", query_idx_2)
        print("------------------------------")
        for idx in [query_idx_1, query_idx_2]:
            cur_query_data = cur_data[idx]
            cur_t = coors[0][idx]
            cur_freq = coors[1][idx]
            query_data.append(cur_query_data[None,...])  # 19 image for each character/class
            query_t.append(cur_t[None, :])
            query_freq.append(cur_freq[None, :])

        support_labels += [i] * support_num
        query_labels += [i] * 2

    support_data = np.concatenate(support_data)
    query_data = np.concatenate(query_data)
    support_t = np.concatenate(support_t)
    support_freq = np.concatenate(support_freq)
    query_t = np.concatenate(query_t)
    query_freq = np.concatenate(query_freq)

    return support_data, support_labels, support_t, support_freq, \
           query_data, query_labels, query_t, query_freq, \
           class_folders


