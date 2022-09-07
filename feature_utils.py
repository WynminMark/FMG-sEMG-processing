# 计算简单的信号时域特征

import numpy as np 

def featureRMS(data):
    'root mean square value'
    return np.sqrt(np.mean(data**2, axis = 0))

def featureMAV(data):
    'mean abs value'
    return np.mean(np.abs(data), axis = 0) 

def featureWL(data):
    'return wave length??'
    # 对每列数据求差分，取绝对值，求和，平均
    # 实质是计算一列信号变化的剧烈程度，变化、跳动越剧烈，数值越大，信号越平缓，数值越小
    return np.sum(np.abs(np.diff(data, axis=0)), axis = 0)/data.shape[0]

def featureZC(data, threshold = 10e-7):
    'calculate zero-crossing rate'
    numOfZC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(1, length):
            diff = data[j, i] - data[j-1, i]
            mult = data[j, i] * data[j-1, i]
            
            if np.abs(diff) > threshold and mult < 0:
                count = count + 1
        numOfZC.append(count/length)
    return np.array(numOfZC)

def featureSSC(data, threshold = 10e-7):
    'return rete of slope sign changes'
    numOfSSC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(2, length):
            diff1 = data[j, i] - data[j-1, i]
            diff2 = data[j-1, i] - data[j-2, i]
            sign  = diff1 * diff2
            
            if sign<0:  # 小于0，斜率符号发生了变化
                if(np.abs(diff1) > threshold or np.abs(diff2) > threshold):
                    count = count + 1
        numOfSSC.append(count/length)
    return np.array(numOfSSC)
