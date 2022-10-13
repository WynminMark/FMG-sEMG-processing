import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

from iFEMG_feature import *
from iFEMGprocessing import read_label


def one_channel_analysis(db_file_path,
                        time_file_path,
                        model_file_path,
                        signal_channel,
                        subject_height,
                        subject_weight,
                        subject_age,
                        subject_gender,
                        subject_name = "test",
                        strength_level = np.NaN):
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
    FMG.signal_segment_label(300)
    sEMG = LabeledsEMGFeature(raw_sEMG, data_time, label, 1223)
    sEMG.signal_segment_label(300)
    # 计算信号特征
    all_feature_df =pd.DataFrame(columns=('subject_name', 
                                    'height(cm)',
                                    'weight(kg)',
                                    'gender',
                                    'age',
                                    'sensor_channel',
                                    'label(kg)', 
                                    'FMG_increase', 
                                    'mav', 
                                    'rms', 
                                    'wave_length', 
                                    'zero_crossing', 
                                    'slope_sign_change', 
                                    'mean_freq', 
                                    'mean_power_freq'))
    temp_FMG_fea = FMG.average_increase()
    temp_mav = sEMG.feature_mav()
    temp_rms = sEMG.feature_rms()
    temp_wl = sEMG.feature_wl()
    temp_zc = sEMG.feature_zc()
    temp_ssc = sEMG.feature_ssc()
    temp_sEMG_freq_fea = sEMG.freq_features()
    temp_len = len(temp_FMG_fea)

    for i in range(temp_len):
        all_feature_df = all_feature_df.append({'subject_name': subject_name,
                                            'height(cm)': subject_height,
                                            'weight(kg)': subject_weight,
                                            'gender': subject_gender,
                                            'age': subject_age,
                                            'sensor_channel': 'bicps_br',
                                            'label(kg)': strength_level,
                                            'FMG_increase': temp_FMG_fea[i],
                                            'mav': temp_mav[i],
                                            'rms': temp_rms[i],
                                            'wave_length': temp_wl[i],
                                            'zero_crossing': temp_zc[i],
                                            'slope_sign_change': temp_ssc[i],
                                            'mean_freq': temp_sEMG_freq_fea[i][0],
                                            'mean_power_freq': temp_sEMG_freq_fea[i][1]}, ignore_index=True)
        pass

    x_data = all_feature_df[['height(cm)', 'weight(kg)', 'gender', 'age', 'label(kg)', 'FMG_increase', 'mav', 'rms', 'wave_length', 'zero_crossing', 'slope_sign_change', 'mean_freq', 'mean_power_freq']].values
    y_data = all_feature_df['label(kg)'].values

    # scaler = preprocessing.StandardScaler().fit(x_data)
    scaler = joblib.load("scaler.save")
    x_to_model = scaler.transform(x_data)

    regression_model = joblib.load(model_file_path)
    y_predict = regression_model.predict(x_to_model)
    print("predicted y: ", y_predict)
    return np.mean(y_predict)
    