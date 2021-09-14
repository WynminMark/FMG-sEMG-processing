# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""
import sqlite3

conn = sqlite3.connect("D:\code\data\iFEMG/grade-0.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
Tables=cursor.fetchall()
print(Tables)










