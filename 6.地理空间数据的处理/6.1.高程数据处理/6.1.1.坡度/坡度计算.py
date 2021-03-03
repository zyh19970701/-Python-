# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:22:10 2020

@author: Admin
"""
from osgeo import gdal
import sys
import numpy as np
import os

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
    DEMFilename = './使用数据/ASTGTM_N31E113C.img'
    slopeFilename = './输出数据/slope_prj.tif'
 
    gdal.AllRegister()
 
    data = gdal.Open(DEMFilename, gdal.GA_ReadOnly)
    if data is None:
        print('Cannot open this file:' + DEMFilename)
        sys.exit(1)
 
    dx = 30  # 分辨率
  
    gridNew = data.ReadAsArray().astype(np.float16)
 
    Sx, Sy = calcFiniteSlopes(gridNew, dx)
    # 坡度计算
    slope = np.arctan(np.sqrt(Sx ** 2 + Sy ** 2)) * 57.29578
    
    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(slopeFilename):
        os.remove(slopeFilename)
    srs1 = data.GetProjection()
    geo1 = data.GetGeoTransform()
    ds1 = driver.Create(slopeFilename, slope.shape[1], slope.shape[0], 1, gdal.GDT_Float32)
    ds1.SetProjection(srs1)
    ds1.SetGeoTransform(geo1)
    band = ds1.GetRasterBand(1)
    band.WriteArray(slope, 0, 0)
    
#    del ds1
    data = None
    projData = None