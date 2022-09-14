# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:02:14 2021

@author: WeimyMark

sEMG信号的基本处理代码
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal

# load data
raw_data = pd.read_table("grade-0.db", sep = ';', header = None)
data_sEMG = raw_data[15].values#7th sEMG
fs = 1200

def band_trap_filter(data, fs, f0):
    Q = 30
    w0 = f0/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    w, h = signal.freqz(b, a)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# function begin
data = band_trap_filter(data_sEMG, fs, 200)
pxx, f = plt.psd(data, NFFT = 512, Fs = fs, Fc = 0, detrend = mlab.detrend_none,
        window = mlab.window_hanning, noverlap = 0, pad_to = None, 
        sides = 'default', scale_by_freq = None, return_line = None)
plt.close()

plt.figure()
plt.plot(f, 10*np.log10(pxx))
plt.show()

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

plt.figure()
plt.plot(diff)
plt.show()

# average power frequency
FSUM = [0]
for i in range(1, N, 1):
    FSUM.append(FSUM[i - 1] + f[i] * pxx[i - 1] * (f[i] - f[i - 1]))
mpf = FSUM[N - 1]/MSUM[N - 1]
power = FSUM[N - 1]
power_time = sum([num*num for num in data])

plt.figure()
plt.plot(MSUM)
plt.plot(FSUM)
plt.show()
"""
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
"""