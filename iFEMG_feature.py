from unittest import result
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

from feature_utils import *


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
        result = []
        rest_value = np.sqrt(np.mean([num**2 for num in self.rest_original_signal], axis = 0))
        for i in self.active_signal_segment:
            i_rms = np.sqrt(np.mean([num**2 for num in i], axis = 0))
            result.append((i_rms - rest_value)/rest_value)
        return result

    def feature_wl(self):
        'return wave length??'
        # 对每列数据求差分，取绝对值，求和，平均
        # 实质是计算一列信号变化的剧烈程度，变化、跳动越剧烈，数值越大，信号越平缓，数值越小
        result = []
        rest_value = np.sum(np.abs(np.diff(self.rest_original_signal, axis=0)), axis = 0)/self.rest_signal_len
        for i in self.active_signal_segment:
            i_wl = np.sum(np.abs(np.diff(i, axis = 0)), axis = 0)/len(i)
            result.append((i_wl - rest_value)/rest_value)
        return result

    def feature_zc(self, threshold = 10e-7):
        'calculate zero-crossing rate'
        result = []
        # 使用函数计算一段信号过零率
        rest_value = calculate_zc(self.rest_original_signal, threshold)
        for i in self.active_signal_segment:
            i_zc = calculate_zc(i, threshold)
            result.append((i_zc - rest_value)/rest_value)
            pass
        return result

    def feature_ssc(self, threshold = 10e-7):
        'return rete of slope sign changes'
        result = []
        rest_value = calculate_ssc(self.rest_original_signal, threshold)
        for i in self.active_signal_segment:
            i_ssc = calculate_ssc(i, threshold)
            result.append((i_ssc - rest_value)/rest_value)
            pass
        return result

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


class AntagonisticMusclesFeature():
    """
    for analysis of antagonistic muscles sEMG signal
    """
    def __init__(self, a_rest_signal, a_active_signal, b_rest_signal, b_active_signal, sample_freq = 1000):
        # init original signal
        self.a_rest = a_rest_signal
        self.a_active = a_active_signal
        self.b_rest = b_rest_signal
        self.b_active = b_active_signal
        self.fs = sample_freq
        pass
    # class end
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