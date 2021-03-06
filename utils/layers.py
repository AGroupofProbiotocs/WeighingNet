# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:44:54 2018

@author: Dragon
"""

import tensorflow as tf
import numpy as np

def conv2d(inputs, kernel_shape, filters, strides=[1,1], padding='SAME', use_bias=True, activation=tf.nn.relu, trainable=True, reuse=False, name='conv'):
    with tf.variable_scope(name, reuse=reuse):
        input_channels = inputs.get_shape()[-1].value
        kernel_shape_tf = [kernel_shape[0], kernel_shape[1], input_channels, filters]
        stride_shape_tf = [1, strides[0], strides[1], 1]
        
        weights = tf.get_variable('weights', kernel_shape_tf, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=trainable)
        bias = tf.get_variable('bias', [filters], initializer=tf.constant_initializer(0.1), trainable=trainable)
        output = tf.nn.conv2d(inputs, weights, stride_shape_tf, padding = padding)
        if use_bias:
            output = tf.nn.bias_add(output, bias)
        
        if activation is not None:
            output = activation(output)

        return output
    
def conv2d_bn(inputs, kernel_shape, filters, strides=[1,1], padding='SAME', activation=tf.nn.relu, is_training=True, trainable=True, reuse=False, name='conv_bn'):
    with tf.variable_scope(name, reuse=reuse):
        input_channels = inputs.get_shape()[-1].value
        kernel_shape_tf = [kernel_shape[0], kernel_shape[1], input_channels, filters]
        stride_shape_tf = [1, strides[0], strides[1], 1]
        He_init = tf.contrib.layers.variance_scaling_initializer()
        # He_init = tf.truncated_normal_initializer(np.sqrt(2/(input_channels*kernel_shape[0]*kernel_shape[1])))
        
        weights = tf.get_variable('weights', kernel_shape_tf, initializer=He_init, trainable=trainable)
#        bias = tf.get_variable('bias', [filters], tf.constant_initializer(0.0), trainable=trainable)
        wx = tf.nn.conv2d(inputs, weights, stride_shape_tf, padding = padding)
#        z = tf.nn.bias_add(z, bias)
        output = tf.contrib.layers.batch_norm(wx, center = True, scale = True, is_training = is_training)
        
        if activation is not None:
            output = activation(output)

        return output

def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)

def global_avg_pool(x, axis=[1, 2], keep_dims=False, name='glb_avg_pool'):
    return tf.reduce_mean(x, axis=axis, keep_dims=keep_dims, name=name)

def global_max_pool(x, axis=[1, 2], keep_dims=False, name='glb_max_pool'):
    return tf.reduce_max(x, axis=axis, keep_dims=keep_dims, name=name)
