import pandas as pd
import numpy as np
import joblib
# private file
from iFEMG_feature import *


def one_channel_analysis(db_file_path,
                        time_file_path,
                        agonist_signal_channel,
                        antagonist_signal_channel,
                        subject_height,
                        subject_weight,
                        subject_age,
                        subject_gender,
                        model_file_path = "OneChannelRegression.pkl",
                        scaler_file_path = "OneChannelScaler.save"):
    """
    输入one channel数据，通过模型输出肌力预测结果

    """
    signal_channel = agonist_signal_channel[0]
    all_feature_df = form_feature_df(db_file_path, time_file_path, signal_channel, subject_height, subject_weight, subject_age, subject_gender, "bicps_br")

    x_data = all_feature_df[['height(cm)', 'weight(kg)', 'gender', 'age', 'FMG_increase', 'mav', 'rms', 'wave_length', 'zero_crossing', 'slope_sign_change', 'mean_freq', 'mean_power_freq']].values
    # y_data = all_feature_df['label(kg)'].values

    print("shape of x_data: ", x_data.shape)
    # scaler = preprocessing.StandardScaler().fit(x_data)
    scaler = joblib.load(scaler_file_path)
    x_to_model = scaler.transform(x_data)
    print("shape of x_to_model: ", x_to_model.shape)

    regression_model = joblib.load(model_file_path)
    y_predict = regression_model.predict(x_to_model)
    print("predicted y: ", y_predict)
    return np.mean(y_predict)


def multi_channel_analysis(db_file_path,
                    time_file_path,
                    model_file_path,
                    signal_channel_list,
                    subject_height,
                    subject_weight,
                    subject_age,
                    subject_gender,
                    subject_name = "test",
                    strength_level = np.NaN):
    
    return 



def form_feature_df(db_file_path,
                    time_file_path,
                    signal_channel,
                    subject_height,
                    subject_weight,
                    subject_age,
                    subject_gender,
                    channel_name,
                    abandon_ms = 300,
                    subject_name = "test",
                    strength_level = np.NaN):
    """
    Return feature_df of one channel iFEMG signal.
    Input: 
        signal_channel: int 1-8
        channel_name: str (muscle name)
    Output:
        unnormalized feature dataframe
    """
    # read data
    raw_data = pd.read_table(db_file_path, sep = ';', header = None)
    label = read_label(time_file_path)
    # read db file
    # row index 0: time
    # row index 1-8: FMG signal
    # row index 9-16: sEMG signal
    # 读取数据array
    data_time = raw_data[0].values
    raw_FMG = raw_data[signal_channel].values
    raw_sEMG = raw_data[signal_channel+8].values

    FMG = LabeledFMGFeature(raw_FMG, data_time, label, 1223)
    FMG.signal_segment_label(abandon_ms)
    sEMG = LabeledsEMGFeature(raw_sEMG, data_time, label, 1223)
    sEMG.signal_segment_label(abandon_ms)
    # 计算信号特征
    temp_FMG_fea = FMG.average_increase()
    temp_mav = sEMG.feature_mav()
    temp_rms = sEMG.feature_rms()
    temp_wl = sEMG.feature_wl()
    temp_zc = sEMG.feature_zc()
    temp_ssc = sEMG.feature_ssc()
    temp_mf, temp_mpf = sEMG.freq_features()
    temp_len = len(temp_FMG_fea)

    subject_name_list = [subject_name for i in range(temp_len)]
    subject_height_list = [subject_height for i in range(temp_len)]
    subject_weight_list = [subject_weight for i in range(temp_len)]
    subject_gender_list = [subject_gender for i in range(temp_len)]
    subject_age_list = [subject_age for i in range(temp_len)]
    channel_name_list = [channel_name for i in range(temp_len)]
    label_list = [strength_level for i in range(temp_len)]

    all_feature_df = pd.DataFrame({'subject_name': subject_name_list,
                                   'height(cm)': subject_height_list,
                                   'weight(kg)': subject_weight_list,
                                   'gender': subject_gender_list,
                                   'age': subject_age_list,
                                   'sensor_channel': channel_name_list,
                                   'label(kg)': label_list,
                                   'FMG_increase': temp_FMG_fea,
                                   'mav': temp_mav,
                                   'rms': temp_rms,
                                   'wave_length': temp_wl,
                                   'zero_crossing': temp_zc,
                                   'slope_sign_change': temp_ssc,
                                   'mean_freq': temp_mf,
                                   'mean_power_freq': temp_mpf})
    return all_feature_df


if __name__ == '__main__':
    subject_arg_input = {"subject_height": 182,
                    "subject_weight": 82,
                    "subject_age": 21,
                    "subject_gender": 1,
                    "subject_name": "Li Peiyang"}
    
    form_feature_df(db_file_path=r"E:\Data\20230424-单人双次iFEMG肌力等级测试\lpy-1\tri-0.db",
                    time_file_path=r"E:\Data\20230424-单人双次iFEMG肌力等级测试\lpy-1\tri-0.txt",
                    signal_channel=1,
                    channel_name="bicps_br",
                    abandon_ms=1000,
                    strength_level=1,
                    **subject_arg_input)