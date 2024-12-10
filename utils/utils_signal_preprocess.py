from collections import Counter
import numpy as np
import datetime
import pandas as pd

def getSampleRate(time_stamp_arr:np.array):
    '''
    计算.db file中数据的每秒采样点数

    Args:
    ----
        * time_stamp_arr: {%Y-%m-%d %H:%M:%S,%f}格式时间array

    '''
    stamp_lst = []
    for i in time_stamp_arr:
        i_stamp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S,%f").timestamp()
        stamp_lst.append(int(i_stamp))
    
    counts = Counter(stamp_lst)
    # 保持原顺序
    result = [(item, counts[item]) for item in counts]
    return result


def getMeanSampleRate(file_path: str)->int:
    '''计算.db文件的平均采样率'''
    df = pd.read_table(file_path, sep=';', header=None)
    sample_per_s = getSampleRate(df[0].values)
    total_sample = 0
    n = len(sample_per_s)
    for i in range(1, n-1):
        total_sample += sample_per_s[i][1]
    return total_sample/(n-2)


def df_time2stamp(df:pd.DataFrame, time_format:str, time_col:str|int, new_col_name:str)->pd.DataFrame:
    '''将df中时间列format格式的时间转换为时间戳，保存为新的一列，返回df。
    使用datetime库实现
    Args:
    ----
        * `df`: dataframe数据
        * `time_format`: 时间列格式, "%Y-%m-%d %H:%M:%S,%f"
        * `time_col`: 时间列的列名
        * `new_col_name`: 保存时间戳的列名
    '''
    stamp_lst = []
    for t in df[time_col].values:
        t_stamp = datetime.datetime.strptime(t, time_format).timestamp()
        stamp_lst.append(t_stamp)
    df[new_col_name] = stamp_lst
    return df


def addSuffix(df:pd.DataFrame, col_name:str)->pd.DataFrame:
    '''给df时间序列中的重复值添加后缀使其唯一，从而方便进行两个df按时间序列的merge，添加新列suffix
    
    Args:
    ----
        * `df`: 需要处理的dataframe
        * `col_name`: 存有时间str的列名
    Features：
    ----
        * 会将col_name转换为datetime格式
        * 添加新列suffix
    '''
    # 将时间列转换为指定格式的 datetime 类型
    df[col_name] = pd.to_datetime(df[col_name], format="%Y-%m-%d %H:%M:%S,%f")
    df['temp_time_100ms'] = df[col_name].dt.floor('100ms')

    # 添加后缀处理重复值
    df['unique_time'] = df.groupby('temp_time_100ms').cumcount() + 1
    df['suffix'] = df['temp_time_100ms'].astype(str) + '_' + df['unique_time'].astype(str)

    # 删除辅助列
    df.drop(columns=['temp_time_100ms', 'unique_time'], inplace=True)
    return df


def getColName(muscle_name:str, fea_lst:list)->list:
    '''获得'''

    return [muscle_name + i for i in fea_lst]