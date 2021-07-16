# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:21:05 2020

@author: Xianglong Zeng
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import *
import numpy as np
import utils.task_generator as tg
import os

FEATURE_DIM = 64
WEIGHING_DIM = 16
CLASS_NUM = 10
SAMPLE_NUM_PER_CLASS = 3
BATCH_NUM_PER_CLASS = 17
TEST_EPISODE = 10000
GPU = 2
DATA_FOLDER = './data'
SHOW_HEAT_MAP = False

class EmbeddingNetwork(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        # self.layer0 = nn.LSTM(151, 151)
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=5,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,FEATURE_DIM,kernel_size=3,padding=1),
                        nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        #input shape: (?, 1, 151, 153)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out # (?, 64, 15, 15)

class WeighingNetwork(nn.Module):
    """docstring for WeighingNetwork"""
    def __init__(self,input_size,hidden_size):
        super(WeighingNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(FEATURE_DIM*2,FEATURE_DIM,kernel_size=3,padding=1),
                        nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(FEATURE_DIM,FEATURE_DIM,kernel_size=3,padding=0),
                        nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.AvgPool2d(3))
        self.fc1 = nn.Sequential(nn.Linear(input_size,hidden_size),
                                 # nn.Dropout(0.5),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, CLASS_NUM)

    def forward(self,x):
        out = self.layer1(x) #(?, 64, 5, 5)
        out = self.layer2(out) #(?, 64, 1, 1)
        out = out.view(-1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM)
        out = torch.mean(out, dim=2, keepdim=False)
        out = out.view(-1, CLASS_NUM*FEATURE_DIM)
        out = self.fc1(out) #(?, 10)
        out = self.fc2(out)  #(?, 5)
        return out

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    test_data_folder = os.path.join(DATA_FOLDER, 'test')
    metatest_gesture_folders = [os.path.join(test_data_folder, gesture_name) \
                                for gesture_name in os.listdir(test_data_folder)]

    # Step 2: init neural networks
    print("init neural networks")

    embedding_module = EmbeddingNetwork()
    weighing_module = WeighingNetwork(CLASS_NUM*FEATURE_DIM, WEIGHING_DIM)

    embedding_module.cuda(GPU)
    weighing_module.cuda(GPU)

    embedding_file = str("./models/gesture_embedding_module_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(FEATURE_DIM) + ".pkl")
    if os.path.exists(embedding_file):
        embedding_module.load_state_dict(torch.load(embedding_file))
        print("load embedding network success")
    weighing_file = str("./models/gesture_weighing_module_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(FEATURE_DIM) + ".pkl")
    if os.path.exists(weighing_file):
        weighing_module.load_state_dict(torch.load(weighing_file))
        print("load weighing network success")

    # test
    print("Testing...")
    accuracies = []

    for i in range(TEST_EPISODE):
        task = tg.GestureTask(metatest_gesture_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS, 1, shuffle_all=False, standardize=False)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="support",shuffle=False)
        test_dataloader = tg.get_data_loader(task,num_per_class=1,split="query",shuffle=True)

        sample_images,sample_labels = sample_dataloader.__iter__().next()
        test_images,test_labels = test_dataloader.__iter__().next()

        embedding_module.eval()
        weighing_module.eval()

        # calculate features
        sample_features = embedding_module(Variable(sample_images).cuda(GPU)) # 25x64x15x15
        test_features = embedding_module(Variable(test_images).cuda(GPU))  # 5x64x15x15

        # calculate probs
        sample_features_ext = sample_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)  #5x25x64x15x15
        test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1) #25x5x64x15x15
        test_features_ext = torch.transpose(test_features_ext, 0, 1)

        weighing_features = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 15, 15)
        logits = weighing_module(weighing_features)
        probs = torch.softmax(logits.data, 1)

        _, predict_labels = torch.max(probs, 1)

        test_labels = test_labels.cuda(GPU)

        rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM)]

        accuracy = np.sum(rewards) / 1.0 / CLASS_NUM
        accuracies.append(accuracy)

    test_accuracy, h = mean_confidence_interval(accuracies)

    print("average_accuracy:", test_accuracy, "h:", h)


if __name__ == '__main__':
    main()
