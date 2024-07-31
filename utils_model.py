import pandas as pd
import itertools

def form_params_grid(params_dict):
    '''
    生成多个参数的所有可能取值组合，并返回一个DataFrame。

    :param params_dict: 字典，键为参数名，值为参数取值列表
    :return: 包含所有参数组合的DataFrame
    '''
    # 生成所有可能的排列组合
    combinations = list(itertools.product(*params_dict.values()))
    
    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=params_dict.keys())
    
    return df