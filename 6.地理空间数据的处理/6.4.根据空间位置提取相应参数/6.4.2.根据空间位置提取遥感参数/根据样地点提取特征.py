# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:31:36 2021

@author: Admin
"""

#根据样地点提取纹理特征
from osgeo import gdal
import numpy as np
import pandas as pd
import os
import osr
from pandas import set_option

class change_coordinate():
    def __init__(self, dataset):
        self.dataset = dataset

    def getSRSPair(self,dataset):
        '''
        获得给定数据的投影参考系和地理参考系
        :param dataset: GDAL地理数据
        :return: 投影参考系和地理参考系
        '''
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs


    def lonlat2geo(self, lon, lat):
        '''
        将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        '''
        prosrs, geosrs = self.getSRSPair(self.dataset)
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        return coords[:2]


    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

    def lonlat2rowcol(self,lon,lat):
        '''
        根据经纬度转行列公式直接转换为行列
        '''
#         tp = self.lonlat2geo(lon,lat)
        geo = self.dataset.GetGeoTransform()
#         row = int((tp[0] -geo[0]) / geo[1]+0.5)
#         col = int((tp[1] - geo[3]) /geo[5]+0.5)
        row = int((lon -geo[0]) / geo[1]+0.5)
        col = int((lat - geo[3]) /geo[5]+0.5)
        
        return row,col
    
class define_window():
    '''
    :param w 定义窗口大小
    :param center_row 中心点行号
    :param center_col 中心点列号
    '''
    def __init__(self,w):
        self.w = w
    def window_upleft_rowcol(self,center_row,center_col):
        upleft_row = center_row - (self.w-1)/2
        upleft_col = center_col - (self.w-1)/2
        return upleft_row,upleft_col

class make_feature_names():
    '''
    根据波段编写特征名称，返回特征名称列表
    '''
    def __init__(self,dataset):
        self.nb = dataset.RasterCount
    def feature(self,feature_list):
        names = []
        for i in range(self.nb):
            for j in feature_list:
                names.append('{}{}{}'.format(j,'_',i))
        return names

if __name__ == '__main__':
    '''
    把图像与坐标放到一个文件夹下
    '''
    img_dir = r'./使用数据'
    out_path=r'./输出数据'
    gdal.AllRegister()
    img = gdal.Open(os.path.join(img_dir,'500_b0_win7_texture.tif'))
    ds = pd.read_excel(os.path.join(img_dir,'point.xls'))
    ns = img.RasterXSize
    nl = img.RasterYSize
    run_change_coordinate = change_coordinate(img)#调用坐标转换函数
    w = 7 #窗口大小
    run_define_window = define_window(w)#调用窗口定义函数

    run_make_feature_names = make_feature_names(img)#调用特征名称函数
    names = [ 'mean_1','variance_1','homogeneity_1','contrast_1','dissimilarity_1','entropy_1','sencond_moment_1','correlation_1',
              ] 

    lon,lat = ds.iloc[:,1].values,ds.iloc[:,2].values
    '''
    定义输出列表
    :all_out输出每个窗口下所有特征的值
    :all_mean输出每个窗口下所有特征的平均值
    :all_std输出每个窗口下所有特征的标准差
    '''
    
    all_out = []
    all_mean = []
    all_std = []
    for i in range(len(lon)):
        ilon,ilat = lon[i],lat[i]
        ix,iy = run_change_coordinate.lonlat2rowcol(ilon,ilat)
        if ix<0 or ix >ns-1 or iy <0 or iy >nl-1:
            print('not in the image: '+str(ds.iat[i,0].value))
        upleft_x,upleft_y = run_define_window.window_upleft_rowcol(ix,iy)
        ref = img.ReadAsArray(int(upleft_x),int(upleft_y),w,w)

        if len(ref.shape) == 3:
            df = np.zeros((w*w,len(names)))         
            for j in range(len(names)):
#                 print(j)
                df[:,j] = list(ref[j].flatten())
            df = pd.DataFrame(df,columns=names)
        else:
            df = pd.DataFrame(ref.flatten())
        description = df.describe()
        df_mean = description.iloc[1,:]
        df_std = description.iloc[2,:]

        all_out.append(df)
        all_mean.append(df_mean)
        all_std.append(df_std)
        
out = pd.concat(all_out)
out_mean = pd.concat(all_mean)
out_std = pd.concat(all_std)
out.to_csv(os.path.join(out_path,'out.csv'))
out_mean.to_csv(os.path.join(out_path,'out_mean.csv'))
out_std.to_csv(os.path.join(out_path,'out_std.csv'))