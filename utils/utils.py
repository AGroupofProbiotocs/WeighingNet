# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 20:22:56 2020

@author: Dragon
"""
import matplotlib.pyplot as plt
import numpy as np
import os,time,cv2
import scipy.io as scio
from scipy import signal
from sklearn.utils import shuffle
import sys
import h5py
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def view_bar(step, total, loss, acc):
    num = step + 1
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    arrow = 0 if num == total else 1
    r = '\rStep:%d/%d [%s%s%s]%d%% - loss: %.4f - acc: %.4f' % \
        (step + 1, total, '■' * rate_num, '▶' * arrow, '-' * (100-rate_num-arrow), rate * 100, loss, acc)
    sys.stdout.write(r)
    sys.stdout.flush()

def generate_img_batch(data_l, data_r, label, batch_size=32, random_shuffle=True):

    # 固定当前随机状态
    # seq_fixed = seq.to_deterministic()

    N = len(label)

    if random_shuffle:
        rand_num = np.int(100 * np.random.random())
        np.random.seed(rand_num)
        np.random.shuffle(data_l)
        np.random.seed(rand_num)
        np.random.shuffle(data_r)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        #判断是否到达最后一个batch，若是则修改该batch的size
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        #将当前图像填装batch
        X_left_batch = data_l[current_index : current_index + current_batch_size]
        X_right_batch = data_r[current_index : current_index + current_batch_size]

        X_batch = {'input_1': X_left_batch, 'input_2': X_right_batch}
        Y_batch = label[current_index : current_index + current_batch_size]

        yield (X_batch, Y_batch)

def spectrogram_double_channel(Path, data_len = 6000, seg_len = 51, noverlap = 12, freq = 2000, sub_freq = 2000,
                               pad_to = 299, shuffle=False, log = False, shift=True, normalization=False,
                               need_coor = False, threshold=None):
    '''
    generate data batch according to the data list(txt file)
    '''
    colum = int(data_len/(int(freq/sub_freq)))
    time_step = int((colum - seg_len) / (seg_len - noverlap)) + 1 #153
    with h5py.File(Path, 'r') as f:
        data_I_all = np.array(f.get('data_I'))
        data_Q_all = np.array(f.get('data_Q'))

    N = len(data_I_all)
    specs = np.zeros((N, pad_to, time_step))
    coors = [np.zeros((N, time_step)), np.zeros((N, pad_to))]

    for i in range(N):
        I = data_I_all[i, 0::int(freq / sub_freq)]
        Q = data_Q_all[i, 0::int(freq / sub_freq)]
        data_IQ = np.zeros(colum,dtype=complex)
        for j in range(colum):
            data_IQ[j] = complex(I[j], Q[j])
        freqs, t, spec = signal.spectrogram(data_IQ, fs=freq, window=('hamming'), nperseg=seg_len,
                                                  noverlap=noverlap, nfft=pad_to, detrend='constant',
                                                  return_onesided=False, scaling='density', axis=-1,
                                                  mode='complex')

        if log:
            spec = 10 * np.log10(abs(spec) + np.spacing(1))
            if threshold is not None:
                spec[spec < threshold] = np.min(spec)
        else:
            spec = abs(spec)

        if shift:
            spec = np.fft.fftshift(spec, 0)
            freqs = np.fft.fftshift(freqs, 0)

        # if normalizadtion:
        #     spec = normalize(spec, 0, 1)
        if need_coor:
            coors[0][i] = t
            coors[1][i] = freqs

        specs[i] = spec

    if shuffle:
        np.random.shuffle(specs)

    if normalization:
        specs = specs/1000 # the max value in our dataset is 980.704

    if need_coor:
        return specs, coors
    else:
        return specs


def spectrogram_single_channel(path_list, colum=6000, NFFT=51, noverlap=12, freq=2000,
                          sub_freq=2000, pad_to=299, shuffle=False, only_real=False):
    '''
    generate data batch according to the data list(txt file)
    '''
    colum = int(colum / (int(freq / sub_freq)))
    step_input_size = int((colum - NFFT) / (NFFT - noverlap)) + 1  # 153
    N = len(path_list)
    if shuffle:
        np.random.shuffle(path_list)

    gesture_spec = np.zeros((N, (pad_to//2)+1, step_input_size))

    for i in range(N):
        data = scio.loadmat(path_list[i])
        data_I = data['d']
        # print(data_I.shape)
        I = data_I[0, 0::int(freq / sub_freq)] #down sample, shape: (6000, )

        freqs, t, spec = signal.spectrogram(I, fs=freq, window=('hamming'), nperseg=NFFT,
                                                  noverlap=noverlap, nfft=pad_to, detrend='constant',
                                                  return_onesided=True, scaling='density', axis=-1,
                                                  mode='complex')

        if only_real:
            spec = 10 * np.log10(abs(spec.real) + np.spacing(1))
        else:
            spec = 10 * np.log10(abs(spec) + np.spacing(1))
        gesture_spec[i] = spec

    return gesture_spec


def count_param(net_inst):
    print('# total parameters: ', sum(param.numel() for param in net_inst.parameters()))
    print('# trainable parameters: ', sum(param.numel() for param in net_inst.parameters() if param.requires_grad))

def normalize(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
        norm_data = np.zeros(data.shape)
    else:
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data


