# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:21:05 2020

@author: Xianglong Zeng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils.utils import *
import numpy as np
import utils.task_generator as tg
import os
import math
import csv
import matplotlib.pyplot as plt
import random

# Hyper Parameters
FEATURE_DIM = 64
WEIGHING_DIM = 16
CLASS_NUM = 10
SAMPLE_NUM_PER_CLASS = 3
BATCH_NUM_PER_CLASS = 17
EPISODE = 100000
TEST_EPISODE = 2000
LEARNING_RATE = 0.001
GPU = 2
LOAD_EXISTED = False
RECORD_FOLDER = './record'
DATA_FOLDER = './data'
SAVE_BEST = True
EARLY_STOP = False
PATIENCE = 30

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
        #input shape: (?, 1, 150, 153)
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
        out = out.view(-1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM)  #(?, 320)
        out = torch.mean(out, dim=2, keepdim=False)
        out = out.view(-1, CLASS_NUM*FEATURE_DIM)
        out = self.fc1(out) #(?, 10)
        out = self.fc2(out)  #(?, 5)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # list gesture folders for data loading
    train_folder = os.path.join(DATA_FOLDER, 'train')
    metatrain_gesture_folders = [os.path.join(train_folder, gesture_name) \
                                 for gesture_name in os.listdir(train_folder)]
    val_folder = os.path.join(DATA_FOLDER, 'val')
    metaval_gesture_folders = [os.path.join(val_folder, gesture_name) \
                               for gesture_name in os.listdir(val_folder)]

    # Step 2: init neural networks
    print("init neural networks")

    embedding_module = EmbeddingNetwork()
    weighing_module = WeighingNetwork(CLASS_NUM*FEATURE_DIM, WEIGHING_DIM)

    print('Embedding network:')
    count_param(embedding_module)
    print('Weighing network:')
    count_param(weighing_module)

    embedding_module.apply(weights_init)
    weighing_module.apply(weights_init)

    embedding_module.cuda(GPU)
    weighing_module.cuda(GPU)

    embedding_module_optim = torch.optim.Adam(embedding_module.parameters(),lr=LEARNING_RATE)
    embedding_module_scheduler = StepLR(embedding_module_optim,step_size=10000,gamma=0.5)
    weighing_module_optim = torch.optim.Adam(weighing_module.parameters(),lr=LEARNING_RATE)
    weighing_module_scheduler = StepLR(weighing_module_optim,step_size=10000,gamma=0.5)

    if LOAD_EXISTED:
        if os.path.exists(str("./models/gesture_embedding_module_concat_fcraw_" + str(CLASS_NUM) +"way_" + str(1) +"shot.pkl")):
            embedding_module.load_state_dict(torch.load(str("./models/gesture_embedding_module_concat_fcraw_" + str(CLASS_NUM) +"way_" + str(1) +"shot.pkl")))
            print("load embedding network success")
        if os.path.exists(str("./models/gesture_weighing_module_concat_fcraw_"+ str(CLASS_NUM) +"way_" + str(1) +"shot.pkl")):
            weighing_module.load_state_dict(torch.load(str("./models/gesture_weighing_module_concat_fcraw_"+ str(CLASS_NUM) +"way_" + str(1) +"shot.pkl")))
            print("load weighing network success")

    # Step 3: build graph
    print("Training...")

    wait = 0
    highest_accuracy = 0.0
    loss_record = []
    acc_record = [0]

    for episode in range(EPISODE):

        embedding_module_scheduler.step(episode)
        weighing_module_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain support samples for compare
        # batch_dataloader is to obtain query samples for training
        task = tg.GestureTask(metatrain_gesture_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS,
                              shuffle_all=False,standardize=False)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="support",shuffle=False)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="query",shuffle=True)


        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()
        # print(samples.shape)
        # print(batch_labels.shape)

        embedding_module.train()
        weighing_module.train()

        # calculate features
        sample_features = embedding_module(Variable(samples).cuda(GPU)) # 25x64x15x15
        batch_features = embedding_module(Variable(batches).cuda(GPU)) # 75x64x15x15

        # calculate probs
        # each batch sample link to every samples to calculate probs
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) # 75x25x64x15x15
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)  # 25x75x64x15x15
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        weighing_features = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,15,15) # 1875x128x15x15
        probs = weighing_module(weighing_features) #95x5

        softmax_cross_entropy = nn.CrossEntropyLoss().cuda(GPU)
        labels = Variable(batch_labels).cuda(GPU)
        loss = softmax_cross_entropy(probs, labels)

        # training
        embedding_module.zero_grad()
        weighing_module.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(embedding_module.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(weighing_module.parameters(),0.5)

        embedding_module_optim.step()
        weighing_module_optim.step()

        if (episode+1)%1000 == 0:
            print("episode:",episode+1,"loss",loss.item())
            loss_record.append(loss.item())

        if (episode+1)%1000 == 0:

            # test
            print("validating...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                task = tg.GestureTask(metaval_gesture_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,1,
                                      shuffle_all=False,standardize=False)
                sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="support",shuffle=False)
                test_dataloader = tg.get_data_loader(task,num_per_class=1,split="query",shuffle=True)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_labels = test_dataloader.__iter__().next()

                embedding_module.eval()
                weighing_module.eval()

                # calculate features
                sample_features = embedding_module(Variable(sample_images).cuda(GPU)) # 25x64x15x15
                test_features = embedding_module(Variable(test_images).cuda(GPU)) # 5x64x15x15

                # calculate probs
                sample_features_ext = sample_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)

                weighing_features = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,15,15)
                logits = weighing_module(weighing_features)
                probs = torch.softmax(logits.data, 1)

                _, predict_labels = torch.max(probs, 1)

                test_labels = test_labels.cuda(GPU)

                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/TEST_EPISODE

            acc_record.append(test_accuracy)

            print("validation accuracy:",test_accuracy)

            if test_accuracy > highest_accuracy:
                print('Validation accuracy increased from %.5f to %.5f' % (highest_accuracy, test_accuracy))
                highest_accuracy = test_accuracy
                if SAVE_BEST:
                    # save networks
                    print("Saving networks for episode:", episode)
                    torch.save(embedding_module.state_dict(), str(
                        "./models/gesture_embedding_module_" + str(CLASS_NUM) + "way_" + str(
                            SAMPLE_NUM_PER_CLASS) + "shot_" + str(FEATURE_DIM) + ".pkl"))
                    torch.save(weighing_module.state_dict(), str(
                        "./models/gesture_weighing_module_" + str(CLASS_NUM) + "way_" + str(
                            SAMPLE_NUM_PER_CLASS) + "shot_" + str(FEATURE_DIM) + ".pkl"))
                if EARLY_STOP:
                    wait = 0
            else:
                print('Validation accuracy did not increased.')
                if EARLY_STOP:
                    wait += 1
                    if wait > PATIENCE:
                        print('Early stop!')
                        break
    if not SAVE_BEST:
        # save networks
        print("Saving networks...")
        torch.save(embedding_module.state_dict(), str(
            "./models/gesture_embedding_module_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot_" + str(FEATURE_DIM) + ".pkl"))
        torch.save(weighing_module.state_dict(), str(
            "./models/gesture_weighing_module_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot_" + str(FEATURE_DIM) + ".pkl"))

    print('highest accuracy in validation is:', highest_accuracy)

    LOSS_RECORD_FOLDER = os.path.join(RECORD_FOLDER, 'loss_record_' + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + 'shot.csv')
    with open(LOSS_RECORD_FOLDER, 'w') as loss_file:
        writer = csv.writer(loss_file)
        writer.writerows([loss_record])

    ACC_RECORD_FOLDER = os.path.join(RECORD_FOLDER, 'acc_record_' + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + 'shot.csv')
    with open(ACC_RECORD_FOLDER, 'w') as acc_file:
        writer = csv.writer(acc_file)
        writer.writerows([acc_record])

    plt.figure()
    plt.plot(loss_record)
    plt.title('loss')
    plt.show()

    plt.figure()
    plt.plot(acc_record)
    plt.title('acc')
    plt.show()

if __name__ == '__main__':
    main()
