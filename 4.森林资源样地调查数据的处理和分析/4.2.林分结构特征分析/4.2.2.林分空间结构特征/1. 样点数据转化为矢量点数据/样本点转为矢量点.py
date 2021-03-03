# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:13:24 2020

@author: 神来伊比
"""

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]#设置全局变量字体为黑体
plt.rcParams["axes.unicode_minus"] = False#设置全局变量正常显示负号


df = pd.read_csv(r'E:\python林业资源\章节20201210\cha4\cha4-2\cha4-2-2\样点数据转化为矢量点数据/使用数据/样地坐标.csv',encoding = 'gb2312')
df['geometry'] = list(zip(df['lon'],df['lat']))

df['geometry'] = df['geometry'].apply(Point)
gpd_df = gpd.GeoDataFrame(df)

gpd_df.plot(color = 'green')
plt.title('样地坐标分布图')
plt.ylabel('lat')
plt.xlabel('lon')
plt.grid()

plt.savefig(r'./输出数据/样地坐标分布图.png',dpi=300)
plt.show()