# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def band_stop_filter():
    pass

def fig_show(data):
    data = np.random.rand(4,2)
    rows = list('1234') #rows categories
    columns = list('MF') #column categories
    fig,ax=plt.subplots()
    #Advance color controls
    ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
    ax.set_xticks(np.arange(0,2)+0.5)
    ax.set_yticks(np.arange(0,4)+0.5)
    # Here we position the tick labels for x and y axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    #Values against each labels
    ax.set_xticklabels(columns,minor=False,fontsize=20)
    ax.set_yticklabels(rows,minor=False,fontsize=20)
    plt.show()
    pass




if __name__ == '__main__':
    raw_data = pdtable_read_db()
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    data_sEMG = raw_data[15].values
    



































