# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:17:51 2021

@author: WeimyMark
"""

with open("D:\code\iFEMG-processing\data\g-0.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        print(line)