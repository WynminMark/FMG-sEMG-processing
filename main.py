# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns 


def read_label():
    # read label.txt to list
    label_raw = []
    label_split = []
    with open("D:\code\data\iFEMG\g-0.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            label_raw.append(line)
    while '' in label_raw:
        label_raw.remove('')   
    for l in label_raw:
        label_split.append(l.split(' '))
    print(len(label_split))
    return label_split


def pdtable_read_db():
    data = pd.read_table("grade-0.db", sep = ';', header = None)
    return data


def band_trap_filter(data, fs, f0):
    Q = 30
    w0 = f0/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    w, h = signal.freqz(b, a)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def fig_show(x, y, title):
    plt.figure()
    if x == None:
        x = np.arange(0, len(y), 1)
        plt.plot(x, y)
    else:
        plt.plot(x, y)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()
    pass


def freq_spec_show(y, fs):
    N = len(y)
    fft_x = np.linspace(0.0, fs/2.0, N//2)
    fft_values = np.fft.fft(y)
    fft_y = 2/N * np.abs(fft_values[0 : N//2])
    
    plt.figure()
    plt.plot(fft_x, fft_y)
    plt.xlabel('freq(Hz)')
    plt.title("frequency specture")
    plt.show()
    pass


def power_spec_show(y, fs):
    N = len(y)
    cor_y = np.correlate(y, y, 'same')
    cor_y_fft = np.fft.fft(cor_y, N)
    ps_cor = np.abs(cor_y_fft)
    ps_cor = ps_cor/np.max(ps_cor)
    
    x_index = np.linspace(0.0, fs/2, N//2)
    
    
    plt.figure()
    plt.plot(x_index, 20*np.log10(ps_cor[:N//2]))
    plt.xlabel('freq(Hz)')
    plt.title("power specture")
    plt.show()
    pass







if __name__ == '__main__':
    raw_data = pdtable_read_db()
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    data_sEMG = raw_data[15].values
    
    fs = 2000
    
    fig_show(None, data_sEMG[:12000], "FMG")
    freq_spec_show(data_sEMG[:12000], fs)
    power_spec_show(data_sEMG[:12000], fs)
    data_sEMG_f = band_trap_filter(data_sEMG, fs, 200)
    fig_show(None, data_sEMG_f[:12000], "FMG")
    freq_spec_show(data_sEMG_f[:12000], fs)
    
    



































