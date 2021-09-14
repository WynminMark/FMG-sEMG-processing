# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:34:49 2021

@author: WeimyMark
"""
# 心电信号滤波模块，包括一般滤波算法、实时滤波
class EEG_filter:

    def __init__(self):
        """一些预设定的参数，或动态数组"""
        self.x_window = []            # 输入动态数组
        self.y_window = []            # 输出动态数组        
        pass

    def reset(self):
        self.x_window = []            # 输入动态数组
        self.y_window = []            # 输出动态数组        

    @staticmethod
    def eeg_filter(data1, data2):
        """心电滤波器设计，包括50Hz陷波和butterworth低通"""
        # data1 为原始信号x，data2 为输出信号y
        lens_data1 = len(data1)
        lens_data2 = len(data2)

        # 滤波器参数
        # 采样率为512Hz时的参数
        # a = [1, -2.62677188325337, 3.61151849824696, -3.00977393361443, 1.73530381308756,
        #     -0.592731739292913, 0.131172714434600, -0.0115850203880053]
        # b = [0.00181588918380624, 0.00793722846596470, 0.0131214763034697, 0.0107787476918359,
        #     0.00843601908020213, 0.0136202669177071, 0.0197416061998655, 0.0215574953836718,
        #     0.0215574953836718, 0.0215574953836718, 0.0215574953836718, 0.0197416061998655,
        #     0.0136202669177071, 0.00843601908020212, 0.0107787476918359, 0.0131214763034696,
        #     0.00793722846596470, 0.00181588918380624]
        # 采样率为224时的参数
        a = [1, 3.60169700100310, 5.86849553560969, 6.46072748090312, 5.86558393000890, 
            4.03507650502189, 1.65862471668638, 0.289501153874726]
        b = [0.107889370669955, 0.611430647199242, 1.61868584046632, 2.87797063231078, 
            4.13725542415524, 5.03662124675236, 5.03662124675236, 4.13725542415524, 
            2.87797063231078, 1.61868584046632, 0.611430647199242, 0.107889370669955]
        lens_a = len(a)
        lens_b = len(b)

        sum1 = 0
        sum2 = 0
        for x in range(1, lens_a):
            if x > lens_data2:
                break
            sum1 = sum1 - a[x] * data2[-x]
            if x >= lens_data2:
                break

        for x in range(0, lens_b):
            if x > lens_data1 - 1:
                break
            sum2 = sum2 + b[x] * data1[-x - 1]
            if x >= lens_data1 - 1:
                break

        return sum1 + sum2


    def real_time_filter(self, data):
        """实时滤波"""
        # data 当前输入
        # 输出滤波结果

        real_time_filter_window_size = 50           # 动态数组窗宽
        self.x_window.append(data)
        a = EEG_filter.eeg_filter(self.x_window, self.y_window)
        self.y_window.append(a)

        if len(self.x_window) > real_time_filter_window_size:
            del self.x_window[0]

        if len(self.y_window) > real_time_filter_window_size:
            del self.y_window[0]

        return a


