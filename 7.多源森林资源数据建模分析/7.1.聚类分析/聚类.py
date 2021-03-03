# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:43:20 2021

@author: Admin
"""

import matplotlib.pyplot as plt  
import numpy as np 
from sklearn import datasets
import time
from sklearn import metrics
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

file_path = r'./使用数据'
data = pd.read_csv(os.path.join(file_path,'样地统计信息.csv'),encoding='utf_8_sig',engine='python')

df = pd.DataFrame(data,columns=['nsp','sprichness','biomass'])

ss_x = StandardScaler().fit(df)
stand_df = ss_x.transform(df)

"""
k-mean聚类
"""
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(stand_df)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签

x0 = ss_x.inverse_transform(stand_df[label_pred == 0])
x1 = ss_x.inverse_transform(stand_df[label_pred == 1])
x2 = ss_x.inverse_transform(stand_df[label_pred == 2])

plt.scatter(x0[:,1], x0[:,2], c="red", marker='o', label='a')  
plt.scatter(x1[:,1], x1[:,2], c="green", marker='*', label='b')  
plt.scatter(x2[:,1], x2[:,2], c="blue", marker='+', label='c')  
plt.xlabel('物种丰富度')  
plt.ylabel('生物量') 
plt.legend(loc=1)  
plt.text(x = -0.01 ,y = 52 , s = r'$Mg·hm^{-2}$',size=8)
# matplotlib.mathtext()
plt.savefig(r'G:\python林业资源\出图/聚类k-mean.png',dpi=300)
plt.show()

center_ids = estimator.cluster_centers_
r1 = pd.Series(label_pred).value_counts()
r2 = pd.DataFrame(center_ids)
r = pd.concat([r2, r1], axis = 1)
r.columns = [u'nsp',u'sprichness',u'biomass'] + [u'类别数目']
print(r)

"""
mean-shift聚类
"""
from sklearn.cluster import MeanShift 
ms = MeanShift(bandwidth=1.4,seeds=None,bin_seeding=False,min_bin_freq=1,cluster_all=True,n_jobs=1)
ms.fit(stand_df)  # 聚类

label_pred = ms.labels_  # 获取聚类标签

x0 = ss_x.inverse_transform(stand_df[label_pred == 0])
x1 = ss_x.inverse_transform(stand_df[label_pred == 1])
x2 = ss_x.inverse_transform(stand_df[label_pred == 2])

plt.scatter(x0[:,1], x0[:,2], c="red", marker='o', label='a')  
plt.scatter(x1[:,1], x1[:,2], c="green", marker='*', label='b')  
plt.scatter(x2[:,1], x2[:,2], c="blue", marker='+', label='c')  
plt.xlabel('物种丰富度')  
plt.ylabel('生物量') 
plt.legend(loc=1)  
plt.text(x = -0.01 ,y = 52 , s = r'$Mg·hm^{-2}$',size=8)
plt.savefig(r'G:\python林业资源\出图/聚类meanshift.png',dpi=300)
plt.show()

center_ids = ms.cluster_centers_
r1 = pd.Series(label_pred).value_counts()
r2 = pd.DataFrame(center_ids)
r = pd.concat([r2, r1], axis = 1)
r.columns = [u'nsp',u'sprichness',u'biomass'] + [u'类别数目']
print(r)

"""
DBSDAN聚类
"""
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.6,  # 邻域半径
min_samples=2,    # 最小样本点数，MinPts
metric='euclidean',
metric_params=None,
algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
leaf_size=20, # balltree,cdtree的参数
p=None, # 
n_jobs=1)
dbscan.fit(stand_df) 

label_pred = dbscan.labels_  # 获取聚类标签

x0 = ss_x.inverse_transform(stand_df[label_pred == 0])
x1 = ss_x.inverse_transform(stand_df[label_pred == 1])
x2 = ss_x.inverse_transform(stand_df[label_pred == 2])

plt.scatter(x0[:,1], x0[:,2], c="red", marker='o', label='a')  
plt.scatter(x1[:,1], x1[:,2], c="green", marker='*', label='b')  
plt.scatter(x2[:,1], x2[:,2], c="blue", marker='+', label='c')  
plt.xlabel('物种丰富度')  
plt.ylabel('生物量') 
plt.legend(loc=1)  
plt.text(x = -0.01 ,y = 24 , s = r'$Mg·hm^{-2}$',size=8)
plt.savefig(r'G:\python林业资源\出图/聚类DBSCAN.png',dpi=300)
plt.show()










