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



def sEMG_analysis(data, fs):
    pxx, f = plt.psd(data, NFFT = 256, Fs = fs, Fc = 0, detrend = mlab.detrend_none,
        window = mlab.window_hanning, noverlap = 0, pad_to = None, 
        sides = 'default', scale_by_freq = None, return_line = None)
    
    # med frequency
    N = len(f)
    MSUM = [0]
    for i in range(1, N, 1):
        MSUM.append(MSUM(i - 1) + Pxx(i - 1) * (f(i) - f(i - 1)))
    
    diff = []
    for i in range(1, N, 1):
        diff.append(MSUM(i) - MSUM(N)/2);

    mfindex = abs(abs(diff)-min(abs(diff))) < 1e-10;
    mf = f(mfindex);
    
    FSUM = zeros(1,len);
    FSUM(1) = 0;
    for i = 2:length(f)
        FSUM(i) = FSUM(i - 1) + f(i) * Pxx(i - 1) * (f(i) - f(i - 1));
    end
    mpf = FSUM(len) / MSUM(len);
    power = FSUM(len);
    
    %% low/med/high band power 
    index15 = sum(f < 15);
    index40 = sum(f <= 40);
    index96 = sum(f <= 96);
    index400 = sum(f <= 400);
    % disp([f(index15), f(index40), f(index96), f(index400)])
    lmp = FSUM(index40) - FSUM(index15);
    mmp = FSUM(index96) - FSUM(index40);
    hmp = FSUM(index400) - FSUM(index96);
    
    %% sEMG integration
    sEMG_abs = abs(rawdata);
    sEMG_integrt = sum(sEMG_abs);
    
    # mean rectification
    m_rect = mean(sEMG_abs);
    
    # rms
    sEMG_rms = rms(rawdata);
    # over-zero counting
    # wave length
    # changes in slope sign
    return mf, lmp, mmp, hmp, power, mpf, sEMG_integrt, m_rect, sEMG_rms



if __name__ == '__main__':
    raw_data = pdtable_read_db()
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    data_sEMG = raw_data[15].values
    
    fs = 1200
    
    fig_show(None, data_sEMG[:12000], "raw sEMG")
    
    fft_x, fft_y = freq_spec(data_sEMG[:12000], fs)
    
    
    
    data_sEMG_f = band_trap_filter(data_sEMG, fs, 200)
    p_x, p_y = power_spec(data_sEMG_f, fs)
    
    #plt.figure()
    pxx, f = plt.psd(data_sEMG_f, NFFT = 256, Fs = 1200, Fc = 0, detrend = mlab.detrend_none,
            window = mlab.window_hanning, noverlap = 0, pad_to = None, 
            sides = 'default', scale_by_freq = None, return_line = None)
    plt.title("psd")
    plt.close()
   # plt.show()
    

    
    



































