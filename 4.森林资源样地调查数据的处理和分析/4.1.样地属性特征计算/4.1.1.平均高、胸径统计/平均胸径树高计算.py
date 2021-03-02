# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:55:23 2020

@author: 神来伊比
"""

import os  
import pandas as pd   
import numpy as np  
import re
file_path = r'./使用数据'       
out_path  = r'./输出数据'  
name_list = os.listdir(file_path)  
#name_list.sort(key = lambda i:int(re.search("([a-z]*)([0-9]*)",i).group(2)))  
out_df = {'plot':[],'平均胸径':[],'平均树高':[]} 
a = 0
for i in name_list:
    out_df['plot'].append(i)
    a+=1
    data = os.path.join(file_path,i)
    df = pd.read_csv(data,encoding ='utf_8_sig')
    count_all = len(df)
    listBins = [0,1,3,5,7,9,11,13,15,17,19,21]
    listLabels = ['0','2','4','6','8','10','12','14','16','18','20']
    df['径阶'] = None
    df['径阶'] = pd.cut(df['胸径'],bins=listBins, labels=listLabels, include_lowest=True)
    high=[]
    area = []
    dbh=[]
    for i in np.unique(df['径阶']):
        h = np.mean(df.loc[df['径阶']==i]['树高'].values)
        
        d = np.square(int(i)/2/100)*3.14
        a = len(df[df['径阶']==i])
        b = np.square(int(i))*a
        dbh.append(b)
        area.append(d)
        c = h*d
        high.append(c)
    H = np.sum(high)/np.sum(area)
    D = np.sqrt(np.sum(dbh)/count_all)
    out_df['平均胸径'].append(D)
    out_df['平均树高'].append(H)
    out_df = pd.DataFrame(out_df)
out_df.to_csv(os.path.join(out_path,'标准地计算.csv'),encoding = 'utf_8_sig',index=False)  
