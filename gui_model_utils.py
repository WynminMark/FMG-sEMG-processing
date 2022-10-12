import pandas as pd
from sklearn import preprocessing
import joblib

from iFEMG_feature import *
from iFEMGprocessing import read_label


def one_channel_analysis(db_file_path, time_file_path, model_file_path):
    # read data
    raw_data = pd.read_table(db_file_path, sep = ';', header = None)
    label = read_label(time_file_path)
    # read db file
    # row index 0: time
    # row index 1-8: FMG signal
    # row index 9-16: sEMG signal
    # 读取数据array
    data_time = raw_data[0].values
    raw_FMG = raw_data[8].values
    raw_sEMG = raw_data[16].values

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
        all_feature_df = all_feature_df.append({'subject_name': 'test',
                                            'label': 'd5',
                                            'FMG_increase': temp_FMG_fea[i],
                                            'mav': temp_mav[i],
                                            'rms': temp_rms[i],
                                            'wave_length': temp_wl[i],
                                            'zero_crossing': temp_zc[i],
                                            'slope_sign_change': temp_ssc[i],
                                            'mean_freq': temp_sEMG_freq_fea[i][0],
                                            'mean_power_freq': temp_sEMG_freq_fea[i][1]}, ignore_index=True)
        pass

    # normalization
    all_fea_norm_df = fea_df_norm(all_feature_df, 'FMG_increase', 'mav', 'rms', 'wave_length', 'zero_crossing', 'slope_sign_change', 'mean_freq', 'mean_power_freq')
    
    '''
    predict muscle strength level
    problem:
        scaler and model
    '''
    # 索引出数据
    d5_data = all_feature_df.loc[all_feature_df.loc[:, 'label'] == 'd5'].values[:, 2:]

    train_data_r = d5_data

    scaler = preprocessing.StandardScaler().fit(train_data_r)
    train_data = scaler.transform(train_data_r)

    regression_model = joblib.load(model_file_path)
    regression_model.predict(train_data)
    
    pass