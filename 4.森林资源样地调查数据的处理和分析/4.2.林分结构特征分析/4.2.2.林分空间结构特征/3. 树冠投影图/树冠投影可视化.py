# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 02:18:49 2017

@author: diyo
plot individual tree crown based on north-south,east-north diameter
"""
import shapefile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def canopy_crow_create(infile,outfile,plotsize = 20):    
    
    csv_file = os.path.basename(infile)
    plotID = csv_file.split('.')[0]  # get from csv
    ds = pd.read_csv(infile,encoding='gb2312')
    npoints = len(ds)  # get from the csv file
    
    coods = ds.iloc[:,5:7].values
    print(coods)
    plt.scatter(coods[:,0],coods[:,1])
    theta = np.arange(-np.pi,np.pi,np.pi/50) #必须构建顺时针方向才可以处理
    sf = shapefile.Writer(outfile)
    sf.autoBalance = 1
    sf.field('PlotID','C','10')
    sf.field('TreeID','C','10')
    sf.field('East_West','N',decimal=8)
    sf.field('North_South','N',decimal=8)
    
    
    for i in range(npoints):
        # read a,b
        a = ds.iloc[i,3]#/2.0 # east-west
        b = ds.iloc[i,4]#/2.0  #get from csv, north-south
        if a > 10 or b >10:
            print(plotID)
        treeID = ds.iloc[i,0] # get from csv
        crown_x = a*np.cos(-theta)+coods[i,0] #必须构建顺时针方向才可以处理
        crown_y = b*np.sin(-theta)+coods[i,1]
        plt.plot(crown_x,crown_y)
        lcrown_xy = np.column_stack((crown_x,crown_y)).tolist()
        sf.poly([lcrown_xy])
        sf.record(plotID,treeID,a,b)
    plt.savefig(r'./输出数据/树冠投影图.png',dpi=300)
if __name__== '__main__':
    file_path = './使用数据'
    out_path = './输出数据'
    csv_file = 'sample_plot.txt'
    infile = os.path.join(file_path,csv_file)
    plotID = csv_file.split('.')[0]  # get from csv
    outfile = os.path.join(out_path,plotID+'_2.shp')
    canopy_crow_create(infile,outfile,20)
    