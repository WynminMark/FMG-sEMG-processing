# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

def read_label():# '2021-09-12 13:12:27.463348 收缩'
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

    
read_label()



































