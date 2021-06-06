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

GPUS = '2'
EPISODES = 100000
TEST_EPISODES = 1000
LEARNING_RATE = 0.0005
LEARNING_RATE_DECAY_STEPS = 1000
LEARNING_RATE_DECAY_RATE = 0.98
HIDDEN_SIZE = 10
QUERY_NUM_PER_CLASS = 19
SUPPORT_NUM_PER_CLASS = 1
CLASS_NUM = 5
PATIENCE = 30
SAVE_BEST  = True
EARLY_STOP = False
DATA_FOLDER = './data'
RECORD_FOLDER = './record'
CKPT_FOLDER = './best_model/'+ str(CLASS_NUM) + 'way_' + str(SUPPORT_NUM_PER_CLASS) + 'shot'


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

    # print(logits.get_shape().as_list())

    probs = tf.nn.softmax(logits, axis=-1)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=similarity, labels=y_true))
    print('successfully build the graph!')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    decayed_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, LEARNING_RATE_DECAY_STEPS,
                                                       LEARNING_RATE_DECAY_RATE, staircase=True)  # lerning rate decay

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)
        grads, variables = zip(*optimizer.compute_gradients(loss))
        grads, global_norm = tf.clip_by_global_norm(grads, 1)
        train_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=tf.global_variables())  # 实例化保存器

    with tf.Session() as sess:
        print('\nStart Training!')
        sess.run(init)
        count_param()

        wait = 0
        best_episode = 0
        highest_accuracy = 0.0
        loss_record = []
        acc_record = []

        for episode in range(EPISODES):

            support_data, query_data, query_labels = generate_meta_data(metatrain_gesture_folders, CLASS_NUM,
                                                                        SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,
                                                                        shuffle=True, normalize=False)

            sess.run(train_op, feed_dict={support_input: support_data, query_input: query_data, labels: query_labels,
                                          is_training: True})

            if (episode + 1) % 1000 == 0:
                train_loss = sess.run(loss, feed_dict={support_input: support_data, query_input: query_data,
                                                       labels: query_labels, is_training: False})
                train_probs = sess.run(probs, feed_dict={support_input: support_data, query_input: query_data,
                                                         is_training: False})
                train_acc = np.sum(np.argmax(train_probs, axis=-1) == query_labels) / QUERY_NUM_PER_CLASS / CLASS_NUM

                loss_record.append(train_loss)
                print("episode:", episode + 1, "loss:", train_loss, " acc:", train_acc)

            if (episode + 1) % 1000 == 0:
                # test
                print("Validating...")
                total_rewards = 0

                for i in range(TEST_EPISODES):
                    support_data, query_data, query_labels = generate_meta_data(metaval_gesture_folders, CLASS_NUM,
                                                                            SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,
                                                                            shuffle=True, normalize=False)
                    val_probs = sess.run(probs, feed_dict={support_input: support_data, query_input: query_data,
                                                           is_training: False})
                    total_rewards += np.sum(np.argmax(val_probs, axis=-1) == query_labels)

                val_accuracy = total_rewards / TEST_EPISODES / CLASS_NUM /QUERY_NUM_PER_CLASS
                acc_record.append(val_accuracy)

                print("validation accuracy:", val_accuracy)

                if val_accuracy > highest_accuracy:
                    best_episode = episode
                    print('Validation accuracy increased from %.5f to %.5f' % (highest_accuracy, val_accuracy))
                    highest_accuracy = val_accuracy
                    if SAVE_BEST:
                        # save networks
                        saver.save(sess, CKPT_FOLDER + '/' + str(CLASS_NUM) + 'way_' + \
                                   str(SUPPORT_NUM_PER_CLASS) + 'shot.ckpt')
                        print("Model saved in file: %s" % CKPT_FOLDER + '/' + str(CLASS_NUM) + 'way_' + \
                              str(SUPPORT_NUM_PER_CLASS) + 'shot.ckpt')
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
            saver.save(sess, CKPT_FOLDER + '/' + str(CLASS_NUM) + 'way_' + \
                               str(SUPPORT_NUM_PER_CLASS) + 'shot.ckpt')
            print("Model saved in file: %s" % CKPT_FOLDER + '/' + str(CLASS_NUM) + 'way_' + \
                  str(SUPPORT_NUM_PER_CLASS) + 'shot.ckpt')

        print('highest accuracy in validation is:', highest_accuracy, ', obtained in episode:', best_episode)


    LOSS_RECORD_FOLDER = os.path.join(RECORD_FOLDER, 'loss_record_' + str(CLASS_NUM) + "way_" + \
                                     str(SUPPORT_NUM_PER_CLASS) + 'shot.csv')
    with open(LOSS_RECORD_FOLDER, 'w') as loss_file:
        writer = csv.writer(loss_file)
        writer.writerows([loss_record])

    ACC_RECORD_FOLDER = os.path.join(RECORD_FOLDER, 'acc_record_' + str(CLASS_NUM) + "way_" + \
                                     str(SUPPORT_NUM_PER_CLASS) + 'shot.csv')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    tf.reset_default_graph()
    main()