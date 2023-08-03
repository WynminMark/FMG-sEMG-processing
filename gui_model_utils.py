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
                    strength_level = np.NaN,
                    signal_sample_freq = 1223):
    """
    Return feature_df of one channel iFEMG signal.
    Input: 
        signal_channel: int 1-8
        channel_name: str (muscle name)
    Output:
        unnormalized feature dataframe
    """
    # read data
    try:
        raw_data = pd.read_table(db_file_path, sep = ';', header = None)
        label = read_label(time_file_path)
    except FileNotFoundError:
        print(f"No such file: {db_file_path, time_file_path}")
        return
    # read db file
    # row index 0: time
    # row index 1-8: FMG signal
    # row index 9-16: sEMG signal
    # 读取数据array
    data_time = raw_data[0].values
    raw_FMG = raw_data[signal_channel].values
    raw_sEMG = raw_data[signal_channel+8].values

    FMG = LabeledFMGFeature(raw_FMG, data_time, label, signal_sample_freq)
    FMG.signal_segment_label(abandon_ms)
    sEMG = LabeledsEMGFeature(raw_sEMG, data_time, label, signal_sample_freq)
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


def FMG_overview(db_file_path: str,
                time_file_path: str,
                signal_channel: int,
                abandon_ms: int = 300,
                signal_sample_freq: int = 1223) -> dict:
    """
    用于描述一段FMG信号的特征，例如平均值，基础值等
    
    * Return:
    ------
        * `dict`{`ave`: signal_ave_value,
            `std`: signal_std_value,
            `initial_pressure_min`: initial_pressure_min,
            `initial_pressure_ave`: initial_pressure_ave,
            `act_ave`: act_ave_list,
            `act_std`: act_std_list,
            `rst_ave`: rst_ave_list,
            `rst_std`:rst_std_list}
    """
    # 读取原始数据
    try:
        raw_data = pd.read_table(db_file_path, sep = ';', header = None)
        label = read_label(time_file_path)
    except FileNotFoundError:
        print(f"No such file: {db_file_path, time_file_path}")
        return
    # read db file
    # row index 0: time
    # row index 1-8: FMG signal
    # row index 9-16: sEMG signal
    # 读取数据array
    data_time = raw_data[0].values
    raw_FMG = raw_data[signal_channel].values
    # 初始化对象
    FMG = LabeledFMGFeature(raw_FMG, data_time, label, signal_sample_freq)
    FMG.signal_segment_label(abandon_ms)
    signal_ave_value, signal_std_value = FMG.get_average_FMG()
    initial_pressure_min, initial_pressure_ave = FMG.get_initial_pressure()
    act_ave_list, act_std_list = FMG.get_avtive_state_FMG()
    rst_ave_list, rst_std_list = FMG.get_rest_state_FMG()
    # print(f"这段信号的平均值是{signal_ave_value}, 标准差为{signal_std_value}, 初始压力最小值为{initial_pressure_min}, 平均值为{initial_pressure_ave}")
    return {"ave": signal_ave_value,
            "std": signal_std_value,
            "initial_pressure_min": initial_pressure_min,
            "initial_pressure_ave": initial_pressure_ave,
            "act_ave": act_ave_list,
            "act_std": act_std_list,
            "rst_ave": rst_ave_list,
            "rst_std":rst_std_list}


def FMG_overview_df(db_file_path: str,
                    time_file_path: str,
                    signal_channel: int,
                    abandon_ms: int = 300,
                    signal_sample_freq: int = 1223) -> pd.DataFrame:
    """
    将FMG_overview的输出转换为dataframe,描述一段FMG信号的特征，例如平均值，基础值等
    
    return
    ------
    * result_df
    """
    result_dict = FMG_overview(db_file_path, time_file_path, signal_channel, abandon_ms, signal_sample_freq)
    ave = result_dict["ave"]
    std = result_dict["std"]
    initial_pressure_min = result_dict["initial_pressure_min"]
    initial_pressure_ave = result_dict["initial_pressure_ave"]
    act_ave_list = result_dict["act_ave"]
    act_std_list = result_dict["act_std"]
    rst_ave_list = result_dict["rst_ave"]
    rst_std_list = result_dict["rst_std"]
    # 获得df长度
    data_len = len(act_ave_list)

    # 获得特征值df
    result_df = pd.DataFrame({"ave": [ave for i in range(data_len)],
                              "std": [std for i in range(data_len)],
                              "initial_pressure_min": [initial_pressure_min for i in range(data_len)],
                              "initial_pressure_ave": [initial_pressure_ave for i in range(data_len)],
                              "act_ave": act_ave_list,
                              "act_std": act_std_list,
                              "rst_ave": rst_ave_list,
                              "rst_std": rst_std_list})
    return result_df


def sEMG_overview_df(db_file_path: str,
                     time_file_path: str,
                     signal_channel: int,
                     abandon_ms: int = 300,
                     signal_sample_freq: int = 1223) -> pd.DataFrame:
    """
    将FMG_overview的输出转换为dataframe,描述一段FMG信号的特征，例如平均值，基础值等
    
    return
    ------
    * result_df
    """
    result_dict = FMG_overview(db_file_path, time_file_path, signal_channel, abandon_ms, signal_sample_freq)
    ave = result_dict["ave"]
    std = result_dict["std"]
    initial_pressure_min = result_dict["initial_pressure_min"]
    initial_pressure_ave = result_dict["initial_pressure_ave"]
    act_ave_list = result_dict["act_ave"]
    act_std_list = result_dict["act_std"]
    rst_ave_list = result_dict["rst_ave"]
    rst_std_list = result_dict["rst_std"]
    # 获得df长度
    data_len = len(act_ave_list)

    # 获得特征值df
    result_df = pd.DataFrame({"ave": [ave for i in range(data_len)],
                              "std": [std for i in range(data_len)],
                              "initial_pressure_min": [initial_pressure_min for i in range(data_len)],
                              "initial_pressure_ave": [initial_pressure_ave for i in range(data_len)],
                              "act_ave": act_ave_list,
                              "act_std": act_std_list,
                              "rst_ave": rst_ave_list,
                              "rst_std": rst_std_list})
    return result_df



def form_sbj_info_df(df_len: int, **subject_info) -> pd.DataFrame:
    """
    把受试者的信息扩展成与特征df相同长度的df
    
    Args:
    ------
    * `df_len`: 需要的df长度
    * `subject_info`: 收拾者的信息，以**{}方式输入
    """
    # 扩展subject_info字典中value的维度，使其与df长度一致，获得新字典
    sbj_info_dict = {}
    for key, value in subject_info.items():
        sbj_info_dict[key] = [value for i in range(df_len)]
        pass
    # 获得受试者个人信息df
    sbj_info_df = pd.DataFrame(sbj_info_dict)
    return sbj_info_df

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