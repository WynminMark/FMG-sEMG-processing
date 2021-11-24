# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:40:28 2021

@author: WeimyMark
"""

import pandas as pd
import seaborn as sns
import iFEMGprocessing as iFEMG


if __name__ == '__main__':
    
    df0 = iFEMG.form_feature_df("D:\code\data\iFEMG_strength_level\grade-10.db", "D:\code\data\iFEMG_strength_level\g-10.txt", "zpk", "0")
    df1 = iFEMG.form_feature_df("D:\code\data\iFEMG_strength_level\grade-11.db", "D:\code\data\iFEMG_strength_level\g-11.txt", "zpk", "1")
    df2 = iFEMG.form_feature_df("D:\code\data\iFEMG_strength_level\grade-12.db", "D:\code\data\iFEMG_strength_level\g-12.txt", "zpk", "2")
    df3 = iFEMG.form_feature_df("D:\code\data\iFEMG_strength_level\grade-13.db", "D:\code\data\iFEMG_strength_level\g-13.txt", "zpk", "3")
    
    features_df = pd.concat([df0, df1, df2, df3], axis = 0, ignore_index = True)
    
    norm1 = iFEMG.fea_df_norm(features_df, 'mf')
    norm2 = iFEMG.fea_df_norm(norm1, 'mpf')
    norm3 = iFEMG.fea_df_norm(norm2, 'power')
    norm4 = iFEMG.fea_df_norm(norm3, 'power_time')
    fea_norm_df = iFEMG.fea_df_norm(norm4, 'FMG_mean')
    
    show_df = pd.DataFrame(columns = ('subject', 'strength_level', 'norm_values', 'fea_name'))
    fea_name_list = ['mf', 'mpf', 'power', 'power_time', 'FMG_mean']
    
    for row in fea_norm_df.itertuples():
        for i in fea_name_list:
            show_df = show_df.append({'subject': row.subject,
                                      'strength_level': row.strength_level,
                                      'norm_values': getattr(row, i),
                                      'fea_name': i}, ignore_index=True)
   
    sns.catplot(x = "fea_name",
                y = "norm_values",
                hue = "strength_level",
                data = show_df,
                kind = 'box')
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    