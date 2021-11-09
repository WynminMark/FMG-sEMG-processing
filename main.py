# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
import time
import datetime
import seaborn as sns


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
    print("label length: ", len(label_split))
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


def fig_show(x, y, title):
    # 快速显示信号
    plt.figure()
    if x:
        plt.plot(x, y)
    else:
        x = np.arange(0, len(y), 1)
        plt.plot(x, y) 
    
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()
    pass


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
    # 用自相关函数的傅里叶变换求信号的功率谱，结果比较像matlab中自带函数的计算结果，
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


def FMG_analysis(data, fs):
    # 处理FMG data，获得该段数据的特征值
    pass


def sEMG_analysis(data, fs):
    # 处理sEMG data，获得该段数据的特征值
    # calculate psd specture
    pxx, f = plt.psd(data, NFFT = 512, Fs = fs, Fc = 0, detrend = mlab.detrend_none,
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


if __name__ == '__main__':
    # read data file.db
    raw_data = pdtable_read_db("grade-0.db")
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    raw_sEMG = raw_data[15].values
    # 滤除不明原因导致的200Hz工频谐波
    data_sEMG = band_trap_filter(raw_sEMG, 1200, 200)
    
    #处理db文件中的ms level时间戳
    t_stamp = [] # 保存数据文件中时间转换的时间戳，精度ms
    for t in data_time:
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        t_stamp.append(ret_stamp)
        
    #处理label.txt中的ms level时间戳
    label = read_label("D:\code\data\iFEMG\g-0.txt")
    label_t_stamp = [] # 保存label文件中时间对应的时间戳，精度ms
    for x in label:
        t = x[0] + " " + x[1]
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        label_t_stamp.append(ret_stamp)
        
    # 比较数据文件和label文件中时间戳，划分肌电活动段
    sEMG_data_set = []
    FMG_data_set = []
    temp_sEMG = []
    temp_FMG = []
    temp_sEMG_r = [] # 静息数据
    temp_FMG_r = []
    rsEMG_data_set = []
    rFMG_data_set = []
    brk_p = 0
    for i in range(len(label) - 1): # 在label时间范围内
        if (label[i][2] == "收缩") and (label[i + 1][2] == "舒张"): # 活动段
            for j in range(len(t_stamp)): # 在整个data长度内搜索，可以优化
                if label_t_stamp[i] <= t_stamp[j] <= label_t_stamp[i + 1]:
                    temp_sEMG.append(data_sEMG[j])
                    temp_FMG.append(data_FMG[j])
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
    
    # 可视化显示活动段和非活动端特征区别
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    