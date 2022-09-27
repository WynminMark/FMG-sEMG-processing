# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:39:51 2021
Description:
    original version 常用基本处理函数库
    calling method in main.py
@author: WeimyMark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
import time
import datetime


def read_label(file_path):
    # 读取motion_guide_gui小程序保存的label文件，col1日期，col2时间，col3动作
    label_raw = []
    label_split = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            label_raw.append(line)
    while '' in label_raw:
        label_raw.remove('')   
    for l in label_raw:
        label_split.append(l.split(' '))
    # print("label length: ", len(label_split))
    return label_split


def pdtable_read_db(file_path):
    # 读取xcd代码保存的.db文件，以;为分隔符，header为None防止第一行数据成为header
    data = pd.read_table(file_path, sep = ';', header = None)
    return data


def band_trap_filter(data, fs, f0):
    # 陷波器
    # fs: sample frequency
    # f0: the frequency to be filtered out
    Q = 30
    w0 = f0/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    w, h = signal.freqz(b, a)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def band_pass_filter(data, fs, fstop1, fstop2):
    b, a = signal.butter(8, [2*fstop1/fs, 2*fstop2/fs], 'bandpass')
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def freq_spec(y, fs):
    # 获得一段信号y的频谱，并return计算结果
    N = len(y)
    fft_x = np.linspace(0.0, fs/2.0, N//2)
    fft_values = np.fft.fft(y)
    fft_y = 2/N * np.abs(fft_values[0 : N//2])

    plt.figure()
    plt.plot(fft_x, fft_y)
    plt.xlabel('freq(Hz)')
    plt.title("frequency specture")
    plt.show()

    return fft_x, fft_y


def power_spec(y, fs):
    '用自相关函数的傅里叶变换求信号的功率谱，结果比较像matlab中自带函数的计算结果，'
    # 和python中自带psd函数的计算结果差异较大
    N = len(y)
    cor_y = np.correlate(y, y, 'same')
    cor_y_fft = np.fft.fft(cor_y, N)
    ps_cor = np.abs(cor_y_fft)
    ps_cor = ps_cor/np.max(ps_cor)
    
    x_index = np.linspace(0.0, fs/2, N//2)
    y_value = 10*np.log10(ps_cor[:N//2])

    plt.figure()
    plt.plot(x_index, ps_cor[:N//2])
    plt.xlabel('freq(Hz)')
    plt.title("self function power specture")
    plt.show()

    return x_index, y_value


def FMG_analysis(FMG, rFMG, fs):
    # 处理FMG data，获得该段数据的特征值
    if len(FMG) != 0 and len(rFMG) != 0:
        FMG_mean = sum(FMG)/len(FMG)
        rFMG_mean = sum(rFMG)/len(rFMG)
        # 计算相对变化，加绝对值避免0级肌力导致变化为小负数
        #relative_FMG_values = abs(FMG_mean - rFMG_mean)/rFMG_mean
        relative_FMG_values = abs(FMG_mean - rFMG_mean)
    else:
        relative_FMG_values = 0
    return relative_FMG_values


def data_segment(raw_data, label):
    # 预处理：滤波；处理时间戳；得到活动段和静息段
    # input:
    # raw_data: dataframe
    # label: list
    # output:
        
    # 读取数据array
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    raw_sEMG = raw_data[15].values
    
    # 滤波
    sEMGf1 = band_pass_filter(raw_sEMG, 1200, 15, 500)
    sEMGf2 = band_trap_filter(sEMGf1, 1200, 200)
    data_sEMG = band_trap_filter(sEMGf2, 1200, 400)
    
    #将db文件中的时间转换为ms level时间戳
    t_stamp = [] # 保存数据文件中时间转换的时间戳，精度ms
    for t in data_time:
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        t_stamp.append(ret_stamp)
        
    #处理label.txt中的ms level时间戳
    # label = read_label("D:\code\data\iFEMG\g-0.txt")
    label_t_stamp = [] # 保存label文件中时间对应的时间戳，精度ms
    for x in label:
        t = x[0] + " " + x[1]
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        label_t_stamp.append(ret_stamp)
        
    # 比较数据文件和label文件中时间戳，划分肌电活动段
    # 储存分割好的数据段
    sEMG_data_set = []
    FMG_data_set = []
    rsEMG_data_set = []
    rFMG_data_set = []
    # 临时一段静息数据和激活数据
    temp_sEMG = []
    temp_FMG = []
    temp_sEMG_r = [] # 静息数据
    temp_FMG_r = []

    for i in range(len(label) - 1): # 在label时间范围内
        if (label[i][2] == "收缩") and (label[i + 1][2] == "舒张"): # 活动段
            for j in range(len(t_stamp)): # 在整个data长度内搜索，可以优化
                if label_t_stamp[i] <= t_stamp[j] <= label_t_stamp[i + 1]:
                    temp_sEMG.append(data_sEMG[j])
                    temp_FMG.append(data_FMG[j])
            if len(temp_FMG) != 0:
                sEMG_data_set.append(temp_sEMG)
                FMG_data_set.append(temp_FMG)
            temp_sEMG = []
            temp_FMG = []
        else: # 非活动段，肌肉静息
            for j in range(len(t_stamp)):
                if label_t_stamp[i] <= t_stamp[j] <= label_t_stamp[i + 1]:
                    temp_sEMG_r.append(data_sEMG[j])
                    temp_FMG_r.append(data_FMG[j])
            rsEMG_data_set.append(temp_sEMG)
            rFMG_data_set.append(temp_FMG)
            temp_sEMG_r = []
            temp_FMG_r = []
    return sEMG_data_set, FMG_data_set, rsEMG_data_set, rFMG_data_set


def sEMG_analysis(data, fs):
    # 处理sEMG data，获得该段数据的特征值
    # calculate psd specture
    pxx, f = plt.psd(data, NFFT = 256, Fs = fs, Fc = 0, detrend = mlab.detrend_none,
        window = mlab.window_hanning, noverlap = 0, pad_to = None, 
        sides = 'default', scale_by_freq = None, return_line = None)
    plt.close()
    
    # med frequency
    N = len(f)
    # calculate (psd curve) integration
    MSUM = [0]
    for i in range(1, N, 1):
        MSUM.append(MSUM[i - 1] + pxx[i - 1] * (f[i] - f[i - 1]))
    
    diff = []
    for i in range(0, N, 1):
        diff.append(MSUM[i] - MSUM[N-1]/2)
    for i in range(N):
        if diff[i] <= 0 and diff[i + 1] >= 0:
            mf_x1= i
            mf_x2 = i + 1
            break
    # linear interpolation based mf calculation
    mf = (f[mf_x1]*diff[mf_x2] - f[mf_x2]*diff[mf_x1])/(diff[mf_x2] - diff[mf_x1])
    
    # average power frequency
    FSUM = [0]
    for i in range(1, N, 1):
        FSUM.append(FSUM[i - 1] + f[i] * pxx[i - 1] * (f[i] - f[i - 1]))
    mpf = FSUM[N - 1]/MSUM[N - 1]
    power = FSUM[N - 1]
    power_time = sum([num*num for num in data])
    return mf, mpf, power, power_time
    #return mf, lmp, mmp, hmp, power, mpf, sEMG_integrt, m_rect, sEMG_rms


def form_feature_df(db_path, label_path, subject, strength_level):
    # 读取数据，
    # 调用预处理函数进行滤波和分段
    # 调用函数，计算特征，返回df类型数据集
    raw_data = pdtable_read_db(db_path)
    label = read_label(label_path)
    sEMG, FMG, rsEMG, rFMG = data_segment(raw_data, label)
    df = pd.DataFrame(columns = ('subject', 'strength_level', 'mf', 'mpf', 'power', 'power_time', 'FMG_mean'))
    
    data_set_num = min([len(FMG), len(rFMG)])
    for i in range(data_set_num):
        mf, mpf, power, power_time = sEMG_analysis(sEMG[i], 1200)
        FMG_mean = FMG_analysis(FMG[i], rFMG[i], 1200)
        df = df.append({'subject': subject,
                        'strength_level': strength_level,
                        'mf': mf,
                        'mpf': mpf,
                        'power': power,
                        'power_time': power_time,
                        'FMG_mean': FMG_mean}, ignore_index=True)
    return df


def fea_df_norm(features_df, col_name):
    # 对feature_df 中的 col_name列进行归一化
    s = (features_df[col_name] - features_df[col_name].min())/(features_df[col_name].max() - features_df[col_name].min())
    #安全删除，如果用del是永久删除
    fea_norm_df = features_df.drop([col_name], axis = 1)
    #把规格化的那一列插入到数组中,最开始的14是我把他插到了第15lie
    fea_norm_df.insert(2, col_name, s)
    return fea_norm_df



