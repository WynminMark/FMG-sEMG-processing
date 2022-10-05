import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import time
import datetime

from feature_utils import *

# 一般数据的处理
# 使用除label以外的方式分割活动段和非活动端
class SignalFeature():
    '1.init original signal. 2.signal segment'
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
        'segment signal using label file.'
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
                print("err: FMG rest average value is 0!")
        return result_list

    # end class
    pass


class LabeledsEMGFeature(LabeledSignalFeature):
    def __init__(self, signal_array, signal_time_array, label, sample_frequency):
        super().__init__(signal_array, signal_time_array, label, sample_frequency)
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
                print("sEMG rest slope sign change rate is 0!")
            pass
        return result_list

    def freq_features(self):
        """
        feature name:
            1.mean frequency
            2.mean power frequency
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
