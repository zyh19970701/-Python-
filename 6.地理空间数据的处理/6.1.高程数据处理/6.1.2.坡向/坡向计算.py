# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:22:12 2020

@author: Admin
"""

from osgeo import gdal
import sys
import numpy as np
import os
import math
from tqdm import tqdm

# 给栅格最外圈加一圈
def assignBCs(elevGrid):
    ny, nx = elevGrid.shape
    Zbc = np.zeros((ny + 2, nx + 2))
    Zbc[1:-1, 1:-1] = elevGrid
 
    Zbc[0, 1:-1] = elevGrid[0, :]
    Zbc[-1, 1:-1] = elevGrid[-1, :]
    Zbc[1:-1, 0] = elevGrid[:, 0]
    Zbc[1:-1, -1] = elevGrid[:, -1]
 
    Zbc[0, 0] = elevGrid[0, 0]
    Zbc[0, -1] = elevGrid[0, -1]
    Zbc[-1, 0] = elevGrid[-1, 0]
    Zbc[-1, -1] = elevGrid[-1, 0]
 
    return Zbc

# 计算dx,dy
def calcFiniteSlopes(elevGrid, dx):
    Zbc = assignBCs(elevGrid)
 
    Sx = (Zbc[1:-1, :-2] - Zbc[1:-1, 2:]) / (2 * dx)  # WE方向
    Sy = (Zbc[2:, 1:-1] - Zbc[:-2, 1:-1]) / (2 * dx)  # NS方向
 
    return Sx, Sy

if __name__ == '__main__':
    DEMFilename = 'd:\查找各种数据资料/Hubei_aster_gdem_30m.tif'
    aspectFilename = 'd:\查找各种数据资料\输出数据/aspect_prj.tif'
 
    gdal.AllRegister()
 
    data = gdal.Open(DEMFilename, gdal.GA_ReadOnly)
    if data is None:
        print('Cannot open this file:' + DEMFilename)
        sys.exit(1)
 
    dx = 30  # 分辨率
  
    gridNew = data.ReadAsArray().astype(np.float16)
 
    Sx, Sy = calcFiniteSlopes(gridNew, dx)
    
    aspect = np.ones([Sx.shape[0], Sx.shape[1]]).astype(np.float32)
    for i in tqdm(range(Sx.shape[0])):
        for j in range(Sy.shape[1]):
            sx = float(Sx[i, j])
            sy = float(Sy[i, j])
            if (sx == 0.0) & (sy == 0.0):
                aspect[i, j] = -1
            elif sx == 0.0:
                if sy > 0.0:
                    aspect[i, j] = 0.0
                else:
                    aspect[i, j] = 180.0
            elif sy == 0.0:
                if sx > 0.0:
                    aspect[i, j] = 90.0
                else:
                    aspect[i, j] = 270.0
            else:
                aspect[i, j] = float(math.atan2(sy, sx) * 57.29578)
                if aspect[i, j] < 0.0:
                    aspect[i, j] = 90.0 - aspect[i, j]
                elif aspect[i, j] > 90.0:
                    aspect[i, j] = 360.0 - aspect[i, j] + 90.0
                else:
                    aspect[i, j] = 90.0 - aspect[i, j]
                    
    # 输出坡度坡向文件
    driver = gdal.GetDriverByName('GTiff')

    if os.path.exists(aspectFilename):
        os.remove(aspectFilename)
    srs2 = data.GetProjection()
    geo2 = data.GetGeoTransform()
    ds2 = driver.Create(aspectFilename, aspect.shape[1], aspect.shape[0], 1, gdal.GDT_Float32)
    ds2.SetProjection(srs2)
    ds2.SetGeoTransform(geo2)
    band = ds2.GetRasterBand(1)
    band.WriteArray(aspect, 0, 0)
 
    del ds2
    data = None
    projData = None