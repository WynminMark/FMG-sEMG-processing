# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import iFEMGprocessing as isp

raw_data = pd.read_table("D:\code\data\iFEMG\grade-0.db", sep = ';', header = None)
label = isp.read_label("D:\code\data\iFEMG\g-0.txt")

sEMG, FMG, rsEMG, rFMG = isp.data_segment(raw_data, label)
df = pd.DataFrame(columns = ('subject', 'strength_level', 'mf', 'mpf', 'power', 'FMG_mean'))

for i in range(len(FMG)):
    mf, mpf, power, power_time = isp.sEMG_analysis(sEMG[i], 1200)
    FMG_mean = isp.FMG_analysis(FMG[i], 1200)
    df = df.append({'subject': 'zpk',
                    'strength_level': '0',
                    'mf': mf,
                    'mpf': mpf,
                    'power': power,
                    'FMG_mean': FMG_mean}, ignore_index=True)


y = sEMG[1]
x = x = np.arange(0, len(y), 1)

plt.plot(x, y)   
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title('abnormal data')
plt.show()

isp.freq_spec(y, 1200)







