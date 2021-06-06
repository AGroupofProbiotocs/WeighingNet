# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:21:05 2020

@author: Dragon
"""

from __future__ import print_function
import csv
import h5py
from utils.utils import *
from utils.task_generator import generate_meta_data
from datetime import datetime
from model import EmbeddingNetwork, WeighingNetwork
from utils.layers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

GPUS = '1'
TEST_EPISODES = 10000
HIDDEN_SIZE = 10
QUERY_NUM_PER_CLASS = 19
SUPPORT_NUM_PER_CLASS = 1
CLASS_NUM = 5
DATA_FOLDER = './data'
CKPT_FOLDER = './best_model/'+ str(CLASS_NUM) + 'way_' + str(SUPPORT_NUM_PER_CLASS) + 'shot'


def main():
    # Step 1: init data folders
    print("init data folders")
    # list gesture folders for data loading
    test_folder = os.path.join(DATA_FOLDER, 'test')
    metatest_gesture_folders = [os.path.join(test_folder, gesture_name) \
                               for gesture_name in os.listdir(test_folder)]

    support_input = tf.placeholder(tf.float32, [SUPPORT_NUM_PER_CLASS*CLASS_NUM, 150, 153, 1], name='support')
    query_input = tf.placeholder(tf.float32, [QUERY_NUM_PER_CLASS*CLASS_NUM, 150, 153, 1], name='query')
    labels = tf.placeholder(tf.int32, [QUERY_NUM_PER_CLASS*CLASS_NUM], name='output')
    is_training = tf.placeholder(tf.bool, name='is_training')

    support_feature = EmbeddingNetwork(support_input)  #5*15*15*64
    query_feature = EmbeddingNetwork(query_input, reuse=True)  #95*15*15*64

    support_feature = tf.reshape(tf.transpose(support_feature, [0,3,1,2]), [-1, 15, 15])
    support_feature = tf.tile(tf.expand_dims(support_feature, 0), [QUERY_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1]) #95*320*15*15
    support_feature = tf.transpose(support_feature, [0,2,3,1])

    weighing_fetures = tf.concat([support_feature, query_feature], axis=-1) #95*15*15*384

    logits = WeighingNetwork(weighing_fetures, hidden_size=HIDDEN_SIZE, output_size=CLASS_NUM)

    probs = tf.nn.softmax(logits, axis=-1)

    accuracies = []

    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(sess, CKPT_FOLDER + '/' + str(CLASS_NUM) + 'way_' + str(SUPPORT_NUM_PER_CLASS) + 'shot.ckpt')

        print('\nStart Testing!')

        for i in range(TEST_EPISODES):
            support_data, query_data, query_labels = generate_meta_data(metatest_gesture_folders, CLASS_NUM,
                                                                    SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,
                                                                        normalize=True)
            val_probs = sess.run(probs, feed_dict={support_input: support_data, query_input: query_data,
                                                   is_training: False})

            rewards = np.argmax(val_probs, axis=-1) == query_labels
            accuracy = np.sum(rewards) / 1.0 / CLASS_NUM / QUERY_NUM_PER_CLASS
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)

        print("average accuracy:", test_accuracy, "h:", h)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    tf.reset_default_graph()
    main()