# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import main


def pdcsv_read():
    data = pd.read_csv("grade-0.db",sep=';')
    print(type(data))

def txt_read():
    with open("grade-0.db") as f:
        f.readlines()

def pdtable_read():
    data = pd.read_table("grade-0.db", sep = ';', header = None)
    print(type(data))
    print(data)

if __name__ == '__main__':










