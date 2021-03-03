# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:59:21 2021

@author: Admin
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'./使用数据\一清数据.csv',encoding='gb2312')
df=df[df['地类'].isin(['111','131','132'])]
"""
面积估计
"""
groupby_df = df.groupby('地类')['样地号'].agg([np.size])
groupby_df['面积成数估计值'] = groupby_df['size'].apply(lambda x:x/np.sum(groupby_df['size']))
groupby_df['面积成数估计值标准差'] = groupby_df['面积成数估计值'].apply(lambda x:np.sqrt(x*(1-x)/(np.sum(groupby_df['size'])-7)))
groupby_df['面积成数估计值标准误'] = groupby_df.apply(lambda x:x['面积成数估计值标准差']/np.sqrt(x['size']),axis=1)
groupby_df['面积估计值'] = groupby_df['面积成数估计值'].apply(lambda x:x*667*np.sum(groupby_df['size']))
groupby_df['面积估计值误差限'] = groupby_df['面积成数估计值标准差'].apply(lambda x:x*2)
groupby_df['抽样精度'] = groupby_df.apply(lambda x:1-2*x['面积成数估计值标准误']/x['面积成数估计值'],axis=1)

groupby_df = groupby_df.rename(columns={'size':'样地数目'})

groupby_df.to_excel(r'./输出数据/面积估计.xlsx')

"""
蓄积估计
"""
grouped_df1 = df.groupby('地类')['活立蓄积'].agg([np.size,np.mean,np.var,np.std])
grouped_df1['面积成数估计值标准误'] = grouped_df1.apply(lambda x:x['std']/np.sqrt(x['size']),axis=1)
grouped_df1['总体总量估计值'] = grouped_df1.apply(lambda x:x['size']*x['mean'],axis=1)
grouped_df1['总体总量估计值误差限'] = grouped_df1.apply(lambda x:x['size']*x['面积成数估计值标准误']*2,axis=1)
grouped_df1['抽样精度'] = grouped_df1.apply(lambda x:1-(2*x['面积成数估计值标准误']/x['mean']),axis=1)

grouped_df1 = grouped_df1.rename(columns={'mean':'样本平均数','size':'样地数目','var':'样地方差','std':'样地标准差'})

groupby_df.to_excel(r'./输出数据/蓄积估计.xlsx')