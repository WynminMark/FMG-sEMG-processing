from collections import Counter
import numpy as np
import datetime
import pandas as pd

def getSamppleRate(time_stamp_arr:np.array):
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


def df_time2stamp(df:pd.DataFrame, time_format:str, time_col:str|int, new_col_name:str)->pd.DataFrame:
    '''将df中时间列format格式的时间转换为时间戳，使用datetime库
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
