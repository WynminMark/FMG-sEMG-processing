import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
from scipy import fft
import numpy as np
import time
import datetime
import pandas as pd
import chardet
import glob
import scipy.io as sio
from sklearn import preprocessing
from typing import Literal

from utils.utils_feature import *


# 一般数据的处理
# 使用除label以外的方式分割活动段和非活动端
class SignalFeature():
    'super class: 1.init original signal. 2.signal segment'
    # signal basic parameters
    rest_original_signal = []   # original signal in 1 row []
    active_original_signal = [] # original signal in 1 row []
    fs = 0

    rest_signal_len = 0     # original signal length
    active_signal_len = 0   # original signal length
    # data segment list, 2D list
    # initiate in signal_segment function
    rest_signal_segment = []
    active_signal_segment = []

    # init function
    def __init__(self, rest_signal, active_signal, sample_freq):
        # input signal: 1D list []
        self.rest_original_signal = rest_signal
        self.active_original_signal = active_signal
        self.fs = sample_freq

        self.rest_signal_len = len(rest_signal)
        self.active_signal_len = len(active_signal)
        pass

    # signal split function
    def signal_segment(self, window_len, step_len):
        'split active/rest signal using sliding window'
        # reset segment list
        self.rest_signal_segment = []
        self.active_signal_segment = []
        # split rest signal
        if self.rest_signal_len < window_len:
            print("Rest signal length is below the window length")
            self.rest_signal_segment.append(self.rest_original_signal)
        else:
            for i in range((self.rest_signal_len - window_len)//step_len + 1):
                self.rest_signal_segment.append(self.rest_original_signal[i*step_len : i*step_len + window_len])
                pass
            pass

        # split active signal
        if self.active_signal_len < window_len:
            print("active signal length is below the window length")
            self.active_signal_segment.append(self.active_original_signal)
        else:
            for i in range((self.active_signal_len - window_len)//step_len + 1):
                self.active_signal_segment.append(self.active_original_signal[i*step_len : i*step_len + window_len])
                pass
            pass
        pass

    def band_pass_filter(self, fstop1, fstop2):
        "band pass filter might induce time delay that need to be dealt with"
        b, a = signal.butter(8, [2*fstop1/self.fs, 2*fstop2/self.fs], 'bandpass')
        filted_data = signal.filtfilt(b, a, self.active_original_signal)
        return filted_data
    # end class
    pass


class FMGFeature(SignalFeature):
    def __init__(self, rest_signal, active_signal, sample_freq):
        super().__init__(rest_signal, active_signal, sample_freq)
        pass

    def rest_mean(self):
        result_list = []
        for i in self.rest_signal_segment:
            result_list.append(sum(i)/len(i))
            pass
        return result_list
    
    def active_mean(self):
        result_list = []
        for i in self.active_signal_segment:
            result_list.append(sum(i)/len(i))
            pass
        return result_list

    def FMG_increase(self):
        'calculate FMG increase: (active - rest) / rest'
        result_list = []
        # calculate average FMG value in rest segment
        rest_mean_FMG = sum(self.rest_original_signal) / self.rest_signal_len
        # calculate results
        for i in self.active_signal_segment:
            result_list.append((sum(i)/len(i) - rest_mean_FMG)/rest_mean_FMG)
            pass
        return result_list
    pass


class sEMGFeature(SignalFeature):
    def __init__(self, rest_signal, active_signal, sample_freq):
        super().__init__(rest_signal, active_signal, sample_freq)
        pass

    def feature_mav(self):
        # calculate average absolute value of rest sEMG
        result_list = []
        rest_mean_sEMG = np.mean([abs(i) for i in self.rest_original_signal])
        for i in self.active_signal_segment:
            result_list.append((np.mean([abs(num) for num in i]) - rest_mean_sEMG)/rest_mean_sEMG)
        return result_list

    def feature_rms(self):
        'root mean square value'
        result_list = []
        rest_value = np.sqrt(np.mean([num**2 for num in self.rest_original_signal], axis = 0))
        for i in self.active_signal_segment:
            i_rms = np.sqrt(np.mean([num**2 for num in i], axis = 0))
            result_list.append((i_rms - rest_value)/rest_value)
        return result_list

    def feature_wl(self):
        'return wave length??'
        # 对每列数据求差分，取绝对值，求和，平均
        # 实质是计算一列信号变化的剧烈程度，变化、跳动越剧烈，数值越大，信号越平缓，数值越小
        result_list = []
        rest_value = np.sum(np.abs(np.diff(self.rest_original_signal, axis=0)), axis = 0)/self.rest_signal_len
        for i in self.active_signal_segment:
            i_wl = np.sum(np.abs(np.diff(i, axis = 0)), axis = 0)/len(i)
            result_list.append((i_wl - rest_value)/rest_value)
        return result_list

    def feature_zc(self, threshold = 10e-7):
        'calculate zero-crossing rate'
        result_list = []
        # 使用函数计算一段信号过零率
        rest_value = calculate_zc(self.rest_original_signal, threshold)
        for i in self.active_signal_segment:
            i_zc = calculate_zc(i, threshold)
            result_list.append((i_zc - rest_value)/rest_value)
            pass
        return result_list

    def feature_ssc(self, threshold = 10e-7):
        'return rete of slope sign changes'
        result_list = []
        rest_value = calculate_ssc(self.rest_original_signal, threshold)
        for i in self.active_signal_segment:
            i_ssc = calculate_ssc(i, threshold)
            result_list.append((i_ssc - rest_value)/rest_value)
            pass
        return result_list

    def freq_features(self):
        """
        feature name:
            mean frequency
            mean power frequency
        """
        result_list = []
        for data in self.active_signal_segment:
            # calculate psd specture
            pxx, f = plt.psd(data, NFFT = 256, Fs = self.fs, Fc = 0, detrend = mlab.detrend_none,
                            window = mlab.window_hanning, noverlap = 0, pad_to = None, 
                            sides = 'default', scale_by_freq = None, return_line = None)
            plt.close()
            '''
            scipy.signal.welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
            '''
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
            
            # average power frequency
            FSUM = [0]
            for i in range(1, N, 1):
                FSUM.append(FSUM[i - 1] + f[i] * pxx[i - 1] * (f[i] - f[i - 1]))
            mpf = FSUM[N - 1]/MSUM[N - 1]
            # power = FSUM[N - 1]
            # power_time = sum([num*num for num in data])
            result_list.append([mf, mpf])
            pass
        return result_list
    # class end
    pass


class AntagonisticFMGFeature():
    '''
    realized based on FMGFeature class
    calculate the difference between FMG signal of antagonistic muscles.
    feature name:
        average_difference
    '''
    def __init__(self, a_rest_signal, a_active_signal, b_rest_signal, b_active_signal, sample_freq = 1223, window_len = 1223, step_len = 500):
        'initialize signal and segment'
        # 初始化两个class，分别是主动肌肉a和拮抗肌肉b
        self.a_FMG = FMGFeature(a_rest_signal, a_active_signal, sample_freq)
        self.b_FMG = FMGFeature(b_rest_signal, b_active_signal, sample_freq)
        # 数据分段
        self.a_FMG.signal_segment(window_len, step_len)
        self.b_FMG.signal_segment(window_len, step_len)
        pass
    
    def average_difference(self):
        'reture mean(agonist) - mean(antagonist)'
        # 计算两块肌肉各自的FMG升高值
        self.a_mean = self.a_FMG.FMG_increase()
        self.b_mean = self.b_FMG.FMG_increase()
        # 转换成array计算差值
        result_array = np.array(self.a_mean) - np.array(self.b_mean)
        return result_array
    # class end
    pass


class AntagonisticsEMGFeature():
    """
    realized based on sEMGFeature class
    for analysis of antagonistic muscles sEMG signal
    """
    def __init__(self, a_rest_signal, a_active_signal, b_rest_signal, b_active_signal, sample_freq = 1223, window_len = 1223, step_len = 500):
        # init original signal
        self.a_sEMG = sEMGFeature(a_rest_signal, a_active_signal, sample_freq)
        self.b_sEMG = sEMGFeature(b_rest_signal, b_active_signal, sample_freq)
        # data sagment
        self.a_sEMG.signal_segment(window_len, step_len)
        self.b_sEMG.signal_segment(window_len, step_len)
        pass

    def mav_difference(self):
        result_a = self.a_sEMG.feature_mav()
        result_b = self.b_sEMG.feature_mav()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array

    def rms_difference(self):
        result_a = self.a_sEMG.feature_rms()
        result_b = self.b_sEMG.feature_rms()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array

    def wl_difference(self):
        result_a = self.a_sEMG.feature_wl()
        result_b = self.b_sEMG.feature_wl()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array

    def zc_difference(self):
        result_a = self.a_sEMG.feature_zc()
        result_b = self.b_sEMG.feature_zc()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array

    def ssc_difference(self):
        result_a = self.a_sEMG.feature_ssc()
        result_b = self.b_sEMG.feature_ssc()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array

    def freq_difference(self):
        result_a = self.a_sEMG.freq_features()
        result_b = self.b_sEMG.freq_features()
        result_array = np.array(result_a) - np.array(result_b)
        return result_array
    # class end
    pass

# 带label数据的处理
class LabeledSignalFeature():
    """
    处理带动作标签的信号
    
    method:
        signal_segment_label: 对信号进行分段并分别存进两个list中
    
    """
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        """
        """
        # 初始化数据，数据的时间戳，label
        self.raw_signal = signal_array
        self.raw_signal_time = signal_time_array
        self.label = label
        self.fs = sample_frequency
        # 储存分段完成的数据
        self.rest_signal_segment = []
        self.active_signal_segment = []
        pass

    def signal_segment_label(self, abandon_ms):
        """
        Segment signal using label file.
        将数据分割成活动段和静息段，分别储存

        Abandon_ms: time abandoned when an action starts.
        """
        # data_time: array
        # label: output of function read_label(file_path)
        # reset segment list

        # reset
        self.rest_signal_segment = []
        self.active_signal_segment = []
        #将db文件中的时间转换为ms level时间戳
        t_stamp = [] # 保存数据文件中时间转换的时间戳，精度ms
        for t in self.raw_signal_time:
            t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f")
            ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
            t_stamp.append(ret_stamp)
        
        #处理label.txt中的ms level时间戳
        # label = read_label("D:\code\data\iFEMG\g-0.txt")
        label_t_stamp = [] # 保存label文件中时间对应的时间戳，精度ms
        for x in self.label:
            t = x[0] + " " + x[1]
            t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
            ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
            label_t_stamp.append(ret_stamp)
        
        # 节约搜索时间，每次从上一次搜索结束处开始搜索
        start_searching_point = 0
        # 临时一段静息数据和激活数据
        temp_active_data = []
        temp_rest_data = []

        for i in range(len(self.label) - 1): # 在label时间范围内
            if (self.label[i][2] == "收缩") and (self.label[i + 1][2] == "舒张"): # 活动段
                for j in range(start_searching_point, len(t_stamp)):
                    if label_t_stamp[i]+abandon_ms <= t_stamp[j] <= label_t_stamp[i + 1]:
                        temp_active_data.append(self.raw_signal[j])
                        start_searching_point = j
                # 把一段数据存进self分段数据列表
                if len(temp_active_data) != 0:
                    self.active_signal_segment.append(temp_active_data)
                # reset list
                temp_active_data = []
            elif (self.label[i][2] == "舒张") and (self.label[i + 1][2] == "收缩"): # 非活动段，肌肉静息
                for j in range(start_searching_point, len(t_stamp)):
                    if label_t_stamp[i]+abandon_ms <= t_stamp[j] <= label_t_stamp[i + 1]:
                        temp_rest_data.append(self.raw_signal[j])
                        start_searching_point = j
                if len(temp_rest_data) != 0:
                    self.rest_signal_segment.append(temp_rest_data)
                temp_rest_data = []
        # 储存特征值/数据段数量
        self.signal_segment_num = min(len(self.active_signal_segment), len(self.rest_signal_segment))
        pass

    def band_pass_filt(self):
        """
        band pass filter and time delay need to be done"""
        pass
    # end class
    pass

class LabeledFMGFeature(LabeledSignalFeature):
    """
    计算带label的FMG信号特征
    
    Method:
        average_increase: 计算相邻活动段与静息段FMG信号的相对差值
    """
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        super().__init__(signal_array, signal_time_array, label, sample_frequency)
        pass

    def average_increase(self):
        """
        计算相邻活动段与静息段FMG信号的相对增加量
        
        (active-rest)/rest
        """
        result_list = []
        for i in range(self.signal_segment_num):
            temp_rest = np.mean(self.rest_signal_segment[i])
            temp_active = np.mean(self.active_signal_segment[i])
            try:
                result_list.append((temp_active - temp_rest)/temp_rest)
            except ZeroDivisionError:
                result_list.append(np.NAN)
                print("err: FMG rest average value is 0!")
        return result_list
    
    def get_average_FMG(self):
        """
        获得整段信号的平均值和标准差

        Return:
            平均值
            标准差
        """
        return np.mean(self.raw_signal), np.std(self.raw_signal)
    
    def get_initial_pressure(self):
        """
        获得初始压力值
        
        Return: 
            第一个放松段的FMG最小值
            第一个放松段的FMG平均值
        """
        return min(self.rest_signal_segment[0]), np.mean(self.rest_signal_segment[0])
    
    def get_avtive_state_FMG(self):
        """
        获得每个活动态FMG的平均值和标准差
        
        Return:
            平均值
            标准差
        """
        ave_FMG = []
        std_FMG = []
        for i in range(self.signal_segment_num):
            ave_FMG.append(np.mean(self.active_signal_segment[i]))
            std_FMG.append(np.std(self.active_signal_segment[i]))

        return ave_FMG, std_FMG
    
    def get_rest_state_FMG(self):
        """
        获得每个放松态FMG的平均值和标准差
        
        Return:
            平均值
            标准差
        """
        ave_FMG = []
        std_FMG = []
        for i in range(self.signal_segment_num):
            ave_FMG.append(np.mean(self.rest_signal_segment[i]))
            std_FMG.append(np.std(self.rest_signal_segment[i]))

        return ave_FMG, std_FMG
    
    # end class
    pass


class LabeledsEMGFeature(LabeledSignalFeature):
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        '''
        初始化滤波器并对sEMG信号进行滤波
        Args:
        ------
        - `signal_array`
        - `signal_time_array`
        - `label`: list, output of read_label()
        - `sample_frequency`
        '''
        super().__init__(signal_array, signal_time_array, label, sample_frequency)
        # 零相位滤波去除信号噪声
        filter1 = NotchFilter(f0=50, fs=self.fs, Q=10)
        filter2 = NotchFilter(f0=150, fs=self.fs, Q=10)
        filter3 = NotchFilter(f0=203.7, fs=self.fs, Q=10)
        filter4 = NotchFilter(f0=250, fs=self.fs, Q=10)
        filter5 = NotchFilter(f0=350, fs=self.fs, Q=10)
        filter6 = NotchFilter(f0=407.4, fs=self.fs, Q=10)
        filter7 = NotchFilter(f0=450, fs=self.fs, Q=10)
        bandpassfilter = ButterFilter(fs = self.fs, fc = [10, 500], order = 8, ftype='bandpass')
        self.raw_signal = bandpassfilter.filt(filter7.filt(filter6.filt(filter5.filt(filter4.filt(filter3.filt(filter2.filt(filter1.filt(signal_array))))))))
        pass
    
    def feature_mav(self, abs_value = False):
        """
        计算mean absolute value of sEMG
        时域幅度特征

        `abs_value`: `True`返回活动态和静息态的绝对值，`False`返回活动态与静息态的相对增加量
        """
        if abs_value:
            result_act = []
            result_rst = []
            for i in range(self.signal_segment_num):
                result_rst.append(np.mean([abs(num) for num in self.rest_signal_segment[i]]))
                result_act.append(np.mean([abs(num) for num in self.active_signal_segment[i]]))
            return result_act, result_rst
        else:
            result_list = []
            for i in range(self.signal_segment_num):
                temp_rest = np.mean([abs(num) for num in self.rest_signal_segment[i]])
                temp_active = np.mean([abs(num) for num in self.active_signal_segment[i]])
                try:
                    result_list.append((temp_active - temp_rest)/temp_rest)
                except ZeroDivisionError:
                    result_list.append(np.NAN)
                    print("err: sEMG rest mav is 0!")
            return result_list
        

    def feature_rms(self, abs_value = False):
        """
        计算root mean square value
        时域幅度特征

        `abs_value`: `True`返回活动态和静息态的绝对值，`False`返回活动态与静息态的相对增加量
        """
        if abs_value:
            result_act = []
            result_rst = []
            for i in range(self.signal_segment_num):
                result_rst.append(np.sqrt(np.mean([num**2 for num in self.rest_signal_segment[i]], axis = 0)))
                result_act.append(np.sqrt(np.mean([num**2 for num in self.active_signal_segment[i]], axis = 0)))
            return result_act, result_rst
        else:
            result_list = []
            for i in range(self.signal_segment_num):
                temp_rest = np.sqrt(np.mean([num**2 for num in self.rest_signal_segment[i]], axis = 0))
                temp_active = np.sqrt(np.mean([num**2 for num in self.active_signal_segment[i]], axis = 0))
                try:
                    result_list.append((temp_active - temp_rest)/temp_rest)
                except ZeroDivisionError:
                    result_list.append(np.NAN)
                    print("sEMG rest rms value is 0!")
            return result_list

    def feature_wl(self, abs_value = False):
        """
        计算wave length
        时域幅度相关特征
        
        `abs_value`: `True`返回活动态和静息态的绝对值，`False`返回活动态与静息态的相对增加量
        """
        # 对每列数据求差分，取绝对值，求和，平均
        # 实质是计算一列信号变化的剧烈程度，变化、跳动越剧烈，数值越大；信号越平缓，数值越小
        if abs_value:
            result_act = []
            result_rst = []
            for i in range(self.signal_segment_num):
                result_rst.append(np.sum(np.abs(np.diff(self.rest_signal_segment[i], axis = 0)), axis = 0)/len(self.rest_signal_segment[i]))
                result_act.append(np.sum(np.abs(np.diff(self.active_signal_segment[i], axis = 0)), axis = 0)/len(self.active_signal_segment[i]))
            return result_act, result_rst
        else:
            result_list = []
            for i in range(self.signal_segment_num):
                temp_rest = np.sum(np.abs(np.diff(self.rest_signal_segment[i], axis = 0)), axis = 0)/len(self.rest_signal_segment[i])
                temp_active = np.sum(np.abs(np.diff(self.active_signal_segment[i], axis = 0)), axis = 0)/len(self.active_signal_segment[i])
                try:
                    result_list.append((temp_active - temp_rest)/temp_rest)
                except ZeroDivisionError:
                    result_list.append(np.NAN)
                    print("sEMG rest wave length is 0!")
            return result_list

    def feature_zc(self, threshold = 10e-7, abs_value = False):
        """
        计算zero-crossing rate
        
        `abs_value`: `True`返回活动态和静息态的绝对值，`False`返回活动态与静息态的相对增加量
        """
        if abs_value:
            result_act = []
            result_rst = []
            for i in range(self.signal_segment_num):
                # 使用函数计算一段信号过零率
                result_rst.append(calculate_zc(self.rest_signal_segment[i], threshold))
                result_act.append(calculate_zc(self.active_signal_segment[i], threshold))
            return result_act, result_rst
        else:
            result_list = []
            for i in range(self.signal_segment_num):
                # 使用函数计算一段信号过零率
                temp_rest = calculate_zc(self.rest_signal_segment[i], threshold)
                temp_active = calculate_zc(self.active_signal_segment[i], threshold)
                try:
                    result_list.append((temp_active - temp_rest)/temp_rest)
                except ZeroDivisionError:
                    result_list.append(np.NAN)
                    print("sEMG rest zero crossing rate is 0!")
                pass
            return result_list

    def feature_ssc(self, threshold = 10e-7, abs_value = False):
        """
        计算slope sign changes rate
        
        `abs_value`: `True`返回活动态和静息态的绝对值，`False`返回活动态与静息态的相对增加量
        """
        if abs_value:
            result_act = []
            result_rst = []
            for i in range(self.signal_segment_num):
                result_rst.append(calculate_ssc(self.rest_signal_segment[i], threshold))
                result_act.append(calculate_ssc(self.active_signal_segment[i], threshold))
            return result_act, result_rst
        else:
            result_list = []
            for i in range(self.signal_segment_num):
                temp_rest = calculate_ssc(self.rest_signal_segment[i], threshold)
                temp_active = calculate_ssc(self.active_signal_segment[i], threshold)
                try:
                    result_list.append((temp_active - temp_rest)/temp_rest)
                except ZeroDivisionError:
                    result_list.append(np.NAN)
                    print("sEMG rest slope sign change rate is 0!")
                pass
            return result_list

    def freq_features(self, method: str = 'welch'):
        '''
        计算活动段的中值频率和平均功率频率
        Args:
        ------
        * `method`: {'welch', 'psd'}
        Outputs:
        ------
        feature name:
            1.mean frequency
            2.mean power frequency
        '''
        mf_list = []
        mpf_list = []
        for index in range(self.signal_segment_num):
            # calculate psd specture
            if method == 'welch':
                f, pxx = signal.welch(self.active_signal_segment[index],
                                      self.fs,
                                      nperseg=len(self.active_signal_segment[index])/8,
                                      noverlap=len(self.active_signal_segment[index])/16)
            elif method == 'psd':
                pxx, f = plt.psd(self.active_signal_segment[index], NFFT = 256, Fs = self.fs, Fc = 0, detrend = mlab.detrend_none,
                            window = mlab.window_hanning, noverlap = 0, pad_to = None, 
                            sides = 'default', scale_by_freq = None, return_line = None)
                plt.close()
            else:
                print(f"Did not calculate mf and mpf")
                return
            '''
            scipy.signal.welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
            '''
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
            
            # average power frequency
            FSUM = [0]
            for i in range(1, N, 1):
                FSUM.append(FSUM[i - 1] + f[i] * pxx[i - 1] * (f[i] - f[i - 1]))
            mpf = FSUM[N - 1]/MSUM[N - 1]
            # power = FSUM[N - 1]
            # power_time = sum([num*num for num in data])
            mf_list.append(mf)
            mpf_list.append(mpf)
            pass
        return mf_list, mpf_list
    
    # end class
    pass


# feature dataframe normalization function
def fea_df_norm(features_df, *col_name):
    #temp_series = []
    fea_norm_df = features_df
    for name in col_name:
        # 对feature_df 中的 col_name列进行归一化
        s = (features_df[name] - features_df[name].min())/(features_df[name].max() - features_df[name].min())
        #安全删除，如果用del是永久删除
        fea_norm_df = fea_norm_df.drop([name], axis = 1)
        fea_norm_df[name] = s
    return fea_norm_df


def read_label(file_path):
    '''
    获得motion_guide_gui.txt保存的label [] 
    '''
    # 读取motion_guide_gui小程序保存的label文件，col1日期，col2时间，col3动作
    label_raw = []
    label_split = []

    with open(file_path, "rb") as f:
        encoding_method = chardet.detect(f.read())["encoding"]

    with open(file_path, "r", encoding=encoding_method) as f:
        for line in f.readlines():
            line = line.strip('\n')
            label_raw.append(line)

    while '' in label_raw:
        label_raw.remove('')   
    for l in label_raw:
        label_split.append(l.split(' '))
    # print("label length: ", len(label_split))
    return label_split


def pdtable_read_db(file_path):
    '''
    读取FMG.db
    '''
    # 读取xcd代码保存的.db文件，以;为分隔符，header为None防止第一行数据成为header
    data = pd.read_table(file_path, sep = ';', header = None)
    return data


def band_trap_filter(data: np.array, fs: int, f0: int):
    """
    基于filtfilt函数实现的零相位陷波器

    Args:
    ------
    * `data`: array like signal.
    * `fs`: sample frequency.
    * `f0`: the frequency to be filtered out.
    """
    Q = 30
    w0 = f0/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    # w, h = signal.freqz(b, a)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def band_pass_filter(data: np.array, fs: int, fstop1: int, fstop2: int, order: int = 4):
    '''
    基于filtfilt和butter实现的零相位带通滤波器

    Args:
    ------
    * `data`: array like signal.
    * `fs`: sample frequency.
    * `fsopt1`/`fstop2`: low/high cut-off frequency
    * `order`: 巴特沃斯滤波器阶数
    '''
    b, a = signal.butter(order, [2*fstop1/fs, 2*fstop2/fs], 'bandpass')
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


class NotchFilter():
    def __init__(self, f0: int = 50, fs: int = 1222, Q: int = 10):
        '''
        iirnotch filter
        Args:
        ------
        * `f0`: center frequency
        * `fs`: sample frequency.
        * `Q`: quality factor
        '''
        self.fs = fs
        self.b, self.a = signal.iirnotch(f0, Q, fs)
        pass

    def show_character(self):
        '''
        显示频率响应曲线
        '''
        w, h = signal.freqz(self.b, self.a, worN=8000)
        plt.figure()
        plt.plot(0.5 * self.fs * w/np.pi, 20 * np.log10(abs(h)), 'b')
        plt.title("Notch filter frequency response")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.show()
        pass

    def filt(self, data: np.array):
        '''
        基于sosfiltfilt的零相位滤波
        Args:
        ------
        * `data`: array like signal.
        '''
        filted_data = signal.sosfiltfilt(signal.tf2sos(self.b, self.a), data)
        return filted_data
    
    # end class
    pass

class BandPassFilter():
    def __init__(self, fs: int, fstop1: int, fstop2: int, order: int = 4):
        '''
        巴特沃斯带通滤波器
        Args:
        ------
        * `fs`: sample frequency.
        * `fsopt1`/`fstop2`: low/high cut-off frequency
        * `order`: 巴特沃斯滤波器阶数
        '''
        self.fs = fs
        self.b, self.a = signal.butter(order, [2*fstop1/fs, 2*fstop2/fs], btype = 'bandpass')
        pass

    def show_character(self):
        '''
        显示滤波器频率响应、相位响应、极点位置
        '''
        # 计算频率响应
        w, h = signal.freqz(self.b, self.a)

        z, p, k = signal.tf2zpk(self.b, self.a)
        # 判断极点的实部是否都小于零
        if np.all(np.real(p) < 0):
            print("stable")
        else:
            print("unstable")

        # 绘制振幅响应曲线
        plt.figure()
        plt.plot(w * self.fs / (2 * np.pi), np.abs(h))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Frequency Response')
        plt.grid(True)

        # 绘制相位响应曲线
        plt.figure()
        plt.plot(w * self.fs / (2 * np.pi), np.angle(h))
        plt.xlabel('Frequency')
        plt.ylabel('Phase')
        plt.title('Phase Response')
        plt.grid(True)

        plt.show()

        # 绘制极点的位置和单位圆
        plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')  # 绘制极点
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='b', label='Unit Circle')  # 绘制单位圆
        plt.axvline(x=0, color='k', linestyle='--')  # 实轴
        plt.axhline(y=0, color='k', linestyle='--')  # 虚轴
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Pole-Zero Plot with Unit Circle')
        plt.axis('equal')  # 设置坐标轴比例相等
        plt.legend()
        plt.grid(True)
        plt.show()
        pass

    def filt(self, data: np.array):
        '''
        基于filtfilt零相位滤波
        Args:
        ------
        * `data`: array like signal.
        '''
        filted_data = signal.filtfilt(self.b, self.a, data)
        return filted_data

    # end class
    pass


class ButterFilter():
    def __init__(self, fs: int = 1200, fc: list|int = ..., order: int = 4, ftype: str = 'lowpass'):
        '''
        Args:
        ------
        * `fc`: critical freq (-3dB point. details in butter). Wn = 2*fc/fs
        * `type`: {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        '''
        self.fs = fs
        self.b, self.a = signal.butter(N = order, Wn = fc, btype = ftype, fs = fs)
        pass

    def show_character(self):
        '''
        显示滤波器频率响应、相位响应、极点位置
        '''
        # 计算频率响应
        w, h = signal.freqz(self.b, self.a)

        z, p, k = signal.tf2zpk(self.b, self.a)
        # 判断极点的实部是否都小于零
        if np.all(np.real(p) < 0):
            print("stable")
        else:
            print("unstable")

        # 绘制振幅响应曲线
        plt.figure()
        plt.plot(w * self.fs / (2 * np.pi), np.abs(h))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Frequency Response')
        plt.grid(True)

        # 绘制相位响应曲线
        plt.figure()
        plt.plot(w * self.fs / (2 * np.pi), np.angle(h))
        plt.xlabel('Frequency')
        plt.ylabel('Phase')
        plt.title('Phase Response')
        plt.grid(True)

        plt.show()

        # 绘制极点的位置和单位圆
        plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')  # 绘制极点
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='b', label='Unit Circle')  # 绘制单位圆
        plt.axvline(x=0, color='k', linestyle='--')  # 实轴
        plt.axhline(y=0, color='k', linestyle='--')  # 虚轴
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Pole-Zero Plot with Unit Circle')
        plt.axis('equal')  # 设置坐标轴比例相等
        plt.legend()
        plt.grid(True)
        plt.show()
        pass

    def filt(self, data: np.array):
        '''
        基于filtfilt零相位滤波
        Args:
        ------
        * `data`: array like signal.
        '''
        filted_data = signal.filtfilt(self.b, self.a, data)
        return filted_data

    # end class
    pass


def freq_spec(y, fs):
    '''获得一段信号y的频谱，并return计算结果'''
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


def show_power_spec(data: np.array, fs: int = 1222):
    '''
    基于welch方法
    nperseg = len(data)/8, noverlap = len(data)/16
    '''
    f, Pxx = signal.welch(data, fs, nperseg=len(data)/8, noverlap=len(data)/16)
    # 绘制功率谱密度图
    # plt.semilogy(f, Pxx)
    plt.plot(f, Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density of the Signal')
    plt.show()

    return f, Pxx


def show_freq_spec(signal: np.array, fs: int = 1222):
    '''
    '''
    fft_result = fft.fft(signal)
    frequencies = fft.fftfreq(len(signal), 1/fs)

    plt.figure()
    plt.plot(frequencies, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    pass


def FMG_analysis(FMG, rFMG, fs):
    '''获得FMG信号的特征
    
    * 计算FMG与rFMG之间的相对变化'''
    # 处理FMG data，获得该段数据的特征值
    if len(FMG) != 0 and len(rFMG) != 0:
        FMG_mean = sum(FMG)/len(FMG)
        rFMG_mean = sum(rFMG)/len(rFMG)
        # 计算相对变化，加绝对值避免0级肌力导致变化为小负数
        #relative_FMG_values = abs(FMG_mean - rFMG_mean)/rFMG_mean
        relative_FMG_values = abs(FMG_mean - rFMG_mean)
    else:
        relative_FMG_values = 0
    return relative_FMG_values


def data_segment(raw_data, label):
    '''
    # 预处理：滤波；处理时间戳；得到活动段和静息段
    # input:
    # raw_data: dataframe
    # label: list
    # output:
    '''
        
    # 读取数据array
    data_time = raw_data[0].values
    data_FMG = raw_data[7].values
    raw_sEMG = raw_data[15].values
    
    # 滤波
    sEMGf1 = band_pass_filter(raw_sEMG, 1200, 15, 500)
    sEMGf2 = band_trap_filter(sEMGf1, 1200, 200)
    data_sEMG = band_trap_filter(sEMGf2, 1200, 400)
    
    #将db文件中的时间转换为ms level时间戳
    t_stamp = [] # 保存数据文件中时间转换的时间戳，精度ms
    for t in data_time:
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        t_stamp.append(ret_stamp)
        
    #处理label.txt中的ms level时间戳
    # label = read_label("D:\code\data\iFEMG\g-0.txt")
    label_t_stamp = [] # 保存label文件中时间对应的时间戳，精度ms
    for x in label:
        t = x[0] + " " + x[1]
        t_array = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        ret_stamp = int(time.mktime(t_array.timetuple()) * 1000 + t_array.microsecond/1000)
        label_t_stamp.append(ret_stamp)
        
    # 比较数据文件和label文件中时间戳，划分肌电活动段
    # 储存分割好的数据段
    sEMG_data_set = []
    FMG_data_set = []
    rsEMG_data_set = []
    rFMG_data_set = []
    # 临时一段静息数据和激活数据
    temp_sEMG = []
    temp_FMG = []
    temp_sEMG_r = [] # 静息数据
    temp_FMG_r = []

    for i in range(len(label) - 1): # 在label时间范围内
        if (label[i][2] == "收缩") and (label[i + 1][2] == "舒张"): # 活动段
            for j in range(len(t_stamp)): # 在整个data长度内搜索，可以优化
                if label_t_stamp[i] <= t_stamp[j] <= label_t_stamp[i + 1]:
                    temp_sEMG.append(data_sEMG[j])
                    temp_FMG.append(data_FMG[j])
            if len(temp_FMG) != 0:
                sEMG_data_set.append(temp_sEMG)
                FMG_data_set.append(temp_FMG)
            temp_sEMG = []
            temp_FMG = []
        else: # 非活动段，肌肉静息
            for j in range(len(t_stamp)):
                if label_t_stamp[i] <= t_stamp[j] <= label_t_stamp[i + 1]:
                    temp_sEMG_r.append(data_sEMG[j])
                    temp_FMG_r.append(data_FMG[j])
            rsEMG_data_set.append(temp_sEMG_r)
            rFMG_data_set.append(temp_FMG_r)
            temp_sEMG_r = []
            temp_FMG_r = []
    return sEMG_data_set, FMG_data_set, rsEMG_data_set, rFMG_data_set


def sEMG_analysis(data, fs):
    # 处理sEMG data，获得该段数据的特征值
    # calculate psd specture
    pxx, f = plt.psd(data, NFFT = 256, Fs = fs, Fc = 0, detrend = mlab.detrend_none,
        window = mlab.window_hanning, noverlap = 0, pad_to = None, 
        sides = 'default', scale_by_freq = None, return_line = None)
    plt.close()
    
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
    
    # average power frequency
    FSUM = [0]
    for i in range(1, N, 1):
        FSUM.append(FSUM[i - 1] + f[i] * pxx[i - 1] * (f[i] - f[i - 1]))
    mpf = FSUM[N - 1]/MSUM[N - 1]
    power = FSUM[N - 1]
    power_time = sum([num*num for num in data])
    return mf, mpf, power, power_time
    #return mf, lmp, mmp, hmp, power, mpf, sEMG_integrt, m_rect, sEMG_rms

'''
该函数在gui_model_utils.py中重新定义了新版本
增加了实验所需的新功能

def form_feature_df(db_path, label_path, subject, strength_level):
    # 读取数据，
    # 调用预处理函数进行滤波和分段
    # 调用函数，计算特征，返回df类型数据集
    raw_data = pdtable_read_db(db_path)
    label = read_label(label_path)
    sEMG, FMG, rsEMG, rFMG = data_segment(raw_data, label)
    df = pd.DataFrame(columns = ('subject', 'strength_level', 'mf', 'mpf', 'power', 'power_time', 'FMG_mean'))
    
    data_set_num = min([len(FMG), len(rFMG)])
    for i in range(data_set_num):
        mf, mpf, power, power_time = sEMG_analysis(sEMG[i], 1200)
        FMG_mean = FMG_analysis(FMG[i], rFMG[i], 1200)
        df = df.append({'subject': subject,
                        'strength_level': strength_level,
                        'mf': mf,
                        'mpf': mpf,
                        'power': power,
                        'power_time': power_time,
                        'FMG_mean': FMG_mean}, ignore_index=True)
    return df
'''

def df_norm(dataframe: pd.DataFrame, col_name: list = [], method: Literal["z_score", "min-max"] = ...) -> pd.DataFrame:
    '''对dataframe的指定列进行归一化
    
    Args:
    ------
        * `dataframe`:
        * `col_name`: 需要进行归一化的列名
        * `method`: 可选`z_score`, `min-max`
    '''
    all_col_name = list(dataframe)  # 获取所有列名
    col_name2drop = [i for i in all_col_name if i not in col_name]  # 列名取差集
    # 暂存不需要进行归一化的数据
    df_not2norm = dataframe[col_name2drop]
    # 需要进行归一化的数据
    df2norm = dataframe[col_name]

    # 根据method进行归一化
    if method == "z_score":
        scaler = preprocessing.StandardScaler().fit_transform(df2norm)
        df_normed = pd.DataFrame(scaler, index = dataframe.index, columns = col_name)
    elif method == "min-max":
        scaler = preprocessing.MinMaxScaler().fit_transform(df2norm)
        df_normed = pd.DataFrame(scaler, index = dataframe.index, columns = col_name)
    else:
        print("ERR! Normalization method dont exist!")
        return
    
    # 对不需归一化和归一化后的df进行合并
    result_df = pd.concat([df_not2norm, df_normed], axis=1)
    return result_df


def df_norm_clinical(dataframe: pd.DataFrame,
                     col_name: list = [],
                     method: Literal["z_score", "min-max"] = ...,
                     train_col_name: str = ("subject_info", "label"),
                     train_col_values: list = [0, 0.5, 1]) -> pd.DataFrame:
    '''
    对dataframe的指定列进行归一化,
    指定训练scaler的数据范围。
    
    Args:
    ------
        * `dataframe`:
        * `col_name`: 需要进行归一化的列名
        * `method`: 可选`z_score`, `min-max`
        * `train_col_name`: 指定df中用于训练scaler的列名
        * `train_col_values`: 指定df中用于训练scaler的列的数值范围
    '''
    all_col_name = list(dataframe)  # 获取所有列名
    col_name2drop = [i for i in all_col_name if i not in col_name]  # 列名取差集
    # 暂存不需要进行归一化的数据
    df_not2norm = dataframe[col_name2drop]
    # 需要进行归一化的数据
    df2norm = dataframe[col_name]
    # 用于训练scaler的df
    train_data = dataframe[dataframe[train_col_name].isin(train_col_values)][col_name]

    # 根据method进行归一化
    if method == "z_score":
        scaler = preprocessing.StandardScaler().fit(train_data).fit_transform(df2norm)
        df_normed = pd.DataFrame(scaler, index = dataframe.index, columns = col_name)
    elif method == "min-max":
        scaler = preprocessing.MinMaxScaler().fit(train_data).fit_transform(df2norm)
        df_normed = pd.DataFrame(scaler, index = dataframe.index, columns = col_name)
    else:
        print("ERR! Normalization method dont exist!")
        return
    
    # 对不需归一化和归一化后的df进行合并
    result_df = pd.concat([df_not2norm, df_normed], axis=1)
    return result_df


def df_norm_with_selfscaler(dataframe: pd.DataFrame, col_name: list = [], self_scaler = ...) -> pd.DataFrame:
    '''
    使用自定义scaler对dataframe的指定列进行归一化
    
    Args:
    ------
        * `dataframe`:
        * `col_name`: 需要进行归一化的列名
        * `scaler`: 自定义scaler

    Outputs:
    ------
        * df
    '''
    all_col_name = list(dataframe)  # 获取所有列名
    col_name2drop = [i for i in all_col_name if i not in col_name]  # 列名取差集
    # 暂存不需要进行归一化的数据
    df_not2norm = dataframe[col_name2drop]
    # 需要进行归一化的数据
    df2norm = dataframe[col_name]

    # 根据method进行归一化
    df_normed = pd.DataFrame(self_scaler.transform(df2norm), index = dataframe.index, columns = col_name)

    # 对不需归一化和归一化后的df进行合并
    result_df = pd.concat([df_not2norm, df_normed], axis=1)
    return result_df
'''
def fea_df_norm(dataframe, col_name = []):
    """
    对feature_df 中的指定列分别进行min-max归一化

    Args:
    ------
        dataframe: 需要进行归一化的dataframe
        col_name: 需要进行归一化的列名
    """
    all_col_name = list(dataframe)  # 获取所有列名
    col_name2drop = [i for i in all_col_name if i not in col_name]  # 列名取差集
    # 暂存不需要进行归一化的数据
    df_not2norm = dataframe[col_name2drop]
    # 需要进行归一化的数据
    df_temp = dataframe[col_name]

    for name in col_name:
        max_value = np.max(df_temp[name])
        min_value = np.min(df_temp[name])
        df_temp[name] = (df_temp[name] - min_value)/(max_value - min_value)
        pass
    result_df = pd.concat([df_not2norm, df_temp], axis=1)
    return result_df
    

def z_score_norm(dataframe, col_name = []):
    """
    z_score 方法归一化
    基于preprocessing.standscaler()实现
    
    输入: dataframe
    输出: dataframe
    """
    all_col_name = list(dataframe)  # 获取所有列名
    col_name2drop = [i for i in all_col_name if i not in col_name]  # 列名取差集
    # 暂存不需要进行归一化的数据
    subject_info_df = dataframe[col_name2drop]
    # 需要进行归一化的数据
    df_temp = dataframe[col_name]
    scaler = preprocessing.StandardScaler().fit_transform(df_temp)
    df_zscore = pd.DataFrame(scaler, index = dataframe.index, columns = col_name)
    # 将不需要归一化的数据和归一化后的数据横向合并
    result_df = pd.concat([subject_info_df, df_zscore], axis = 1)
    return result_df
'''

def df_save_csv(dataframe, filename):
    '''
    把dataframe存到文件路径filename处
    
    覆写检测，避免损失之前的数据
    不保存index
    '''
    # Use this function to search for any files which match your filename
    files_present = glob.glob(filename)
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        dataframe.to_csv(filename, index = False, date_format="%Y-%m-%d %H:%M:%S,%f")
        print(f"File {filename} saved!")
    else:
        print(f"WARNING: File {filename} NOT saved (same file already exists!)")
    pass


def df_merge_colname(df: pd.DataFrame) -> pd.DataFrame:
    '''
    把df多重列名字符串拼接合并成单列名
    '''
    df_copy = df.copy() # 复制dataframe防止修改原df，导致多次运行结果不一致
    old_col_name_list = list(df_copy)
    new_col_name_list = []
    for name in old_col_name_list:
        new_col_name_list.append("_".join(name))
        pass

    df_copy.columns = new_col_name_list
    
    return df_copy


def statistical_outlier_filter(data, thrs = 3):
    """
    基于平均值和标准差滤除离群值
    
    thrs = 3 默认平均值上下三倍标准差范围外为离群值
    """
    # 计算平均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 筛选离群值
    filtered_data = data[(data > mean - thrs * std) & (data < mean + thrs * std)]
    return filtered_data


def box_outlier_filter(data, thrs = 1.5):
    """
    基于箱线图滤除离群值
    
    thrs = 1.5 默认上下四分位数两侧距离超过1.5倍上下四分位数距离为离群值
    """
    # 绘制箱线图
    fig, ax = plt.subplots()
    ax.boxplot(data)

    # 筛选离群值
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    filtered_data = [x for x in data if q1 - thrs * iqr < x < q3 + thrs * iqr]
    return filtered_data


def db2mat(folder_path, file_name = []) -> None:
    """
    将folder_path路径中xcd-8+8channel.db文件中的数据转存为.mat文件并保存在原路径中, 方便Matlab进行读取操作；
    可以自动忽略已转换和不存在的文件；
    Args
    ------
    * `folder_path`: 存放db数据的文件夹
    * `file_name` = ["bi-0", "bi-05", "bi-1", "bi-2"]|["tri-0", "tri-05", "tri-1"]
    """
    for name in file_name:
        # 获得保存mat路径
        mat_name = name.replace("-", "_")
        mat_path = folder_path + "\\" + mat_name + ".mat"

        # Use this function to search for any files which match your filename
        files_present = glob.glob(mat_path)
        # if no matching files, write to csv, if there are matching files, print statement
        if not files_present:
            # 读取文件路径中的数据
            db_path = folder_path + "\\" + name + ".db"
            try:
                raw_data = pd.read_table(db_path, sep = ';', header = None)
            except FileNotFoundError:
                print(f"{db_path} doesn't exist!")
                continue
            # 提取第1-16路数据，转换为float64类型
            data = raw_data.iloc[:, 1:17].values.astype(np.float64)

            sio.savemat(mat_path, {mat_name: data})
            print(f"File {mat_path} saved!")
        else:
            print(f"WARNING: File {mat_path} NOT saved (same file already exists!)")
            pass
        pass
    return



if __name__ == '__main__':
    pass