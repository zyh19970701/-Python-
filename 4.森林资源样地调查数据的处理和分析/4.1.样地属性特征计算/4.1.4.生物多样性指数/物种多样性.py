# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:32:26 2020

@author: 神来伊比
"""

import pandas as pd
import numpy as np
import os

file_path = r'./使用数据'       
out_path  = r'./输出数据'
name_list = os.listdir(file_path) 

dict_df1 = {'plot':[],'树种':[],'数量':[]}

for i in name_list:
    file = os.path.join(file_path,i)
    df = pd.read_csv(file,encoding='utf_8_sig')
    tree_names = list(df['树种'].unique())
    for j in tree_names:
        num = len(df.loc[df['树种']==j])
        dict_df1['plot'].append(i)
        dict_df1['树种'].append(j)
        dict_df1['数量'].append(num)
out_df1 = pd.DataFrame(dict_df1)      
dict_df = {'plot':[],'shannon':[],'pielou_shannon':[],'simpson':[],'pielou_simpson':[]}
for i in out_df1['plot'].unique():
    clip = out_df1.loc[out_df1['plot']==i]
    tree_list = clip['树种'].unique()
    dict_df['plot'].append(i)
    
    H = []
    D = []
    for l in tree_list:
        N = np.sum(clip['数量'].values)
        S = len(np.unique(clip['树种']))
        n = np.sum(clip[clip['树种']==l]['数量'].values)
        pi = n/N       
        h = pi*np.log(pi)
        H.append(h)
        d = pi*pi
        D.append(d)
    shannon = -np.sum(H)
    simpson = 1-np.sum(D)
    pieloush = shannon/np.log(S)
    pielousi = simpson/(1-1/S)
    dict_df['shannon'].append(shannon)
    dict_df['simpson'].append(simpson)
    dict_df['pielou_shannon'].append(pieloush)  
    dict_df['pielou_simpson'].append(pielousi)
    
data = pd.DataFrame(dict_df)
data.to_csv(os.path.join(out_path,'diversity.csv'),encoding='utf_8_sig')