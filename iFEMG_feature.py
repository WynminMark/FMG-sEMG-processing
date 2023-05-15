import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
import numpy as np
import time
import datetime
import pandas as pd
import chardet
import glob

from feature_utils import *


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
    'labeled signal feature extraction'
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
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
        '''
        Segment signal using label file.
        Abandon_ms: time abandoned when an action starts.
        '''
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
        "band pass filter and time delay need to be done"
        pass
    # end class
    pass

class LabeledFMGFeature(LabeledSignalFeature):
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        super().__init__(signal_array, signal_time_array, label, sample_frequency)
        pass

    def average_increase(self):
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

    # end class
    pass


class LabeledsEMGFeature(LabeledSignalFeature):
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        super().__init__(signal_array, signal_time_array, label, sample_frequency)
        # 零相位滤波去除基线漂移
        self.raw_signal = band_pass_filter(signal_array, self.fs, 10, 500)
        pass
    
    def feature_mav(self):
        'increase% of mean absolute value of sEMG'
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

    def feature_rms(self):
        'increase% of root mean square value'
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

    def feature_wl(self):
        'increase% of wave length'
        # 对每列数据求差分，取绝对值，求和，平均
        # 实质是计算一列信号变化的剧烈程度，变化、跳动越剧烈，数值越大，信号越平缓，数值越小
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

    def feature_zc(self, threshold = 10e-7):
        'increase% of zero-crossing rate'
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

    def feature_ssc(self, threshold = 10e-7):
        'return rete of slope sign changes'
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

    def freq_features(self):
        """
        feature name:
            1.mean frequency
            2.mean power frequency
        """
        mf_list = []
        mpf_list = []
        for index in range(self.signal_segment_num):
            # calculate psd specture
            pxx, f = plt.psd(self.active_signal_segment[index], NFFT = 256, Fs = self.fs, Fc = 0, detrend = mlab.detrend_none,
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


def band_trap_filter(data, fs, f0):
    '''
    # 陷波器

    # fs: sample frequency
    # f0: the frequency to be filtered out
    '''
    Q = 30
    w0 = f0/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    # w, h = signal.freqz(b, a)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def band_pass_filter(data, fs, fstop1, fstop2):
    """
    zero-phase filter
    """
    b, a = signal.butter(8, [2*fstop1/fs, 2*fstop2/fs], 'bandpass')
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def freq_spec(y, fs):
    # 获得一段信号y的频谱，并return计算结果
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


def power_spec(y, fs):
    '''
    用自相关函数的傅里叶变换求信号的功率谱
    结果比较像matlab中自带函数的计算结果
    '''
    # 和python中自带psd函数的计算结果差异较大
    N = len(y)
    cor_y = np.correlate(y, y, 'same')
    cor_y_fft = np.fft.fft(cor_y, N)
    ps_cor = np.abs(cor_y_fft)
    ps_cor = ps_cor/np.max(ps_cor)
    
    x_index = np.linspace(0.0, fs/2, N//2)
    y_value = 10*np.log10(ps_cor[:N//2])

    plt.figure()
    plt.plot(x_index, ps_cor[:N//2])
    plt.xlabel('freq(Hz)')
    plt.title("self function power specture")
    plt.show()

    return x_index, y_value


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

def fea_df_norm(features_df, *col_name):
    # 对feature_df 中的 col_name列进行归一化
    for name in col_name:
        s = (features_df[name] - features_df[name].min())/(features_df[name].max() - features_df[name].min())
        #安全删除，如果用del是永久删除
        fea_norm_df = features_df.drop([name], axis = 1)
        #把规格化的那一列插入到数组中,最开始的14是我把他插到了第15lie
        fea_norm_df.insert(2, name, s)
    return fea_norm_df

def df_save_csv(dataframe, filename):
    '''把dataframe存到文件路径filename处
    
    覆写检测，避免损失之前的数据'''
    # Use this function to search for any files which match your filename
    files_present = glob.glob(filename)
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        dataframe.to_csv(filename)
        print('Done!')
    else:
        print('WARNING: This file already exists!' )
    pass

