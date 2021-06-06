import tensorflow as tf
import numpy as np
from utils.layers import conv2d, conv2d_bn, global_avg_pool
import math

def EmbeddingNetwork(inputs, is_training=True, reuse=False, name='EmbeddingNetwork'):
    '''
    Built the Embedding Network.
        
    '''
    with tf.variable_scope(name, reuse=reuse):
        x_1 = conv2d_bn(inputs, [5, 5], 64, activation=tf.nn.relu, is_training=is_training, padding='VALID', name='conv_1')
        x_1 = tf.nn.max_pool(x_1, [1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

        x_2 = conv2d_bn(x_1, [3, 3], 64, activation=tf.nn.relu, is_training=is_training, padding='VALID', name='conv_2')
        x_2 = tf.nn.max_pool(x_2, [1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

        x_3 = conv2d_bn(x_2, [3, 3], 64, activation=tf.nn.relu, is_training=is_training, padding='SAME', name='conv_3')

        x_4 = conv2d_bn(x_3, [3, 3], 64, activation=tf.nn.relu, is_training=is_training, padding='SAME', name='conv_4')

    return x_4

def WeighingNetwork(inputs, hidden_size, output_size, is_training=True, reuse=False, name='WeighingNetwork'):
    '''
    Built the Weighing Network.

    '''
    with tf.variable_scope(name, reuse=reuse):
        x_1 = conv2d_bn(inputs, [3, 3], 64, activation=tf.nn.relu, is_training=is_training, padding='SAME', name='conv_1')
        x_1 = tf.nn.max_pool(x_1, [1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

        x_2 = conv2d_bn(x_1, [3, 3], 64, activation=tf.nn.relu, is_training=is_training, padding='VALID', name='conv_2')
        x_2 = global_avg_pool(x_2)

        x_3 = tf.layers.dense(x_2, hidden_size, activation=tf.nn.relu, name='dense_1')

        x_4 = tf.layers.dense(x_3, output_size, name='dense_2')

    return x_4



