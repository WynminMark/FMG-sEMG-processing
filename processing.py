# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:34:50 2021

@author: WeimyMark
"""
'''
数据信号处理，执行数据采集后的算法处理
'''

from framework.utils.CounterBuilder import CounterBuilder
from bizcom_BlueToothDataAcquire.Msgs import DACompletedMsg, DataProcessMsg
from collections import deque
# from bizcom_BlueToothDataAcquire.algorithm.Smooth import smooth
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from framework.msgcore.MsgSubscribe import MsgSubscribe
from bizcom_BlueToothDataAcquire.algorithm.EEG_filter import *
from bizcom_BlueToothDataAcquire.algorithm.PPG_filter import *
from bizcom_BlueToothDataAcquire.algorithm.R_detect import *

SMOOTH_WINDOWSIZE = 10
PROCESS_DATA_LENGTH = 200

class DataProcessor:

    def __init__(self):
        self.preDaCompleteMsg : DACompletedMsg  = None
        self.ppgfilter1 : PPG_filter = PPG_filter()
        self.ppgfilter2 : PPG_filter = PPG_filter()
        self.ppgfilter3 : PPG_filter = PPG_filter()
        self.ppgfilter4 : PPG_filter = PPG_filter()
        self.ppgfilter5 : PPG_filter = PPG_filter()        
        self.eegfitler6 : EEG_filter = EEG_filter()
        self.rdetect6 : R_detect = R_detect()
        self.counter : CounterBuilder = CounterBuilder()
        pass

    @MsgSubscribe()
    def onDataProcessMsg(self, dpMsg:DataProcessMsg):
        daCompletedMsg = dpMsg.daMsg
        self.counter.addCounter("Received DataProcessMsg")

        if( self.preDaCompleteMsg != None and self.preDaCompleteMsg.StartTime != daCompletedMsg.StartTime):
            if(self.ppgfilter1 != None):
                self.ppgfilter1.reset()
            if(self.ppgfilter2 != None):
                self.ppgfilter2.reset()
            if(self.ppgfilter3 != None):
                self.ppgfilter3.reset()
            if(self.ppgfilter4 != None):
                self.ppgfilter4.reset()
            if(self.ppgfilter5 != None):
                self.ppgfilter5.reset()
            if(self.eegfitler6 != None):
                self.eegfitler6.reset()
            if(self.rdetect6 != None):
                self.rdetect6.reset()            

        self.preDaCompleteMsg = daCompletedMsg
        daCompletedMsg.ChannelValue[11] = self.ppgfilter1.real_time_filter(daCompletedMsg.ChannelValue[1])
        daCompletedMsg.ChannelValue[12] = self.ppgfilter2.real_time_filter(daCompletedMsg.ChannelValue[2])
        daCompletedMsg.ChannelValue[13] = self.ppgfilter3.real_time_filter(daCompletedMsg.ChannelValue[3])
        daCompletedMsg.ChannelValue[14] = self.ppgfilter4.real_time_filter(daCompletedMsg.ChannelValue[4])
        daCompletedMsg.ChannelValue[15] = self.ppgfilter5.real_time_filter(daCompletedMsg.ChannelValue[5])
        daCompletedMsg.ChannelValue[16] = self.eegfitler6.real_time_filter(daCompletedMsg.ChannelValue[6])
        r = self.rdetect6.detect_real_time(daCompletedMsg.ChannelValue[16])
        if r :
            daCompletedMsg.ChannelValue[17] = f"{r[0]},{r[1]}"
        else:
            daCompletedMsg.ChannelValue[17] = f""
