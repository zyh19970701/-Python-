# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:17:22 2021

@author: Admin
"""

#引入相关函数库
import pandas as pd
import numpy as np
import glob
import os
from osgeo import gdal,gdal_array
#import glob
from tqdm import tqdm
import multiprocessing as mp
import argparse
import time
import random
from skimage.feature import greycomatrix, greycoprops

def getminmaxatpercent(im,min_perc=0.02,max_perc=0.98):
    im_fla = im.flatten().astype('int')
    minv = im.min()
    maxv = im.max()
    hist = np.zeros(maxv.astype('int')+1)
#     hist = hist.astype('int')

#    histgram
    for i in np.arange(np.size(im_fla)):
        hist[im_fla[i]] = hist[im_fla[i]]+1
    
    hist = hist/np.sum(hist)
    
    hist_cum = np.cumsum(hist)
    
    for i in np.arange(len(hist_cum)):
        if hist_cum[i] >= min_perc:
            minv = i
            break
    for i in np.arange(len(hist_cum)-1,0,-1):    
        if hist_cum[i] <= max_perc:
            maxv = i
            break
    return minv,maxv
#拉伸函数
def stretch(im,minv,maxv,ngrey = 256,gamma = None):
    im = np.where(im >minv,im,minv)
    im = np.where(im <maxv,im,maxv)
    tmp = (im - minv)*1.0/(maxv - minv)
    if gamma == None:
        im = np.uint8(tmp*(ngrey-1)+0.5)
    else:
        im = np.uint8((ngrey-1)*np.power(tmp,1.0/gamma))
    return im
##########
# 根据指定的窗口参数，计算灰度共生矩阵6各纹理参数
# windbuf 图像数据
# step 灰度共生矩阵参数 
#
#######

def glcm_feas(winbuf,step = 1,angle = np.pi/4):
    try:
        comatrix = greycomatrix(winbuf,[step],[angle],256,symmetric=True,normed= True)
        corr = greycoprops(comatrix,prop = 'correlation')
        con = greycoprops(comatrix,prop = 'contrast')
        dissimi = greycoprops(comatrix,prop = 'dissimilarity')
        homo = greycoprops(comatrix,prop = 'homogeneity')
        asm = greycoprops(comatrix,prop = 'ASM')
        energy = greycoprops(comatrix,prop = 'energy')
        return (corr,con,dissimi,homo,asm,energy)
    except:
        return (0,0,0,0,0,0)
    
'''
类似greycoprops函数，计算灰度共生矩阵的熵、均值、标准差
prop :{entropy,mean,variance}
'''   

def glcm_other_feas(P,prop='entropy'):
    (num_level,num_level2,num_dist,num_angle) = P.shape
    if num_level !=num_level2:
        raise ValueError('num_level and num_level2 must be equal!')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive')
    if num_angle<=0:
        raise ValueError('num_angle must be positive')
        
    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums
    
    # create weights for specified property
    
    if prop == 'entropy':
        mask0 = P < 1e-15
        P[mask0] = 1
        results = np.apply_over_axes(np.sum,(-P*np.log(P)),axes=(0,1))[0,0]
    elif prop=='mean':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        mean_j = np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]
        
#        results=(mean_i+mean_j) /2
        results = mean_i
    elif prop=='variance':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]
        std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                           axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                           axes=(0, 1))[0, 0])
#        results= (std_i +std_j) /2
        results = std_i
    
    return results
"""
imgbuf: 是原始数据往外扩大winsize/2的大小，返回的是减去了winsize/2后
winsize
step, angle

""" 
def paddedbuf(im,winsize):
    off = int(winsize/2)
    ny,nx = im.shape
    padded = np.zeros((ny + 2*off,nx+2*off),dtype = np.uint8)
    padded[off:off+ny,off:off+nx] = im
    
    up = padded[off+1:2*off+1,off:-off][::-1]
    padded[:off,off:-off] = up
    
    bottom = padded[ny-1:off+ny-1,off:-off][::-1]
    padded[off+ny:,off:-off] = bottom
    
    left = padded[:,off+1:2*off+1][:,::-1]
    padded[:,0:off]= left
    
    right = padded[:,nx-1:nx-1+off][:,::-1]
    padded[:,nx+off:] = right
    return padded

def glcm_by_win_scan(imgbuf,winsize=3,step = 1, angle = np.pi/4,level = 256):
    nx = imgbuf.shape[1]
    ny = imgbuf.shape[0]
    hwin = int(winsize/2)
    
    outx = nx - winsize +1
    outy = ny - winsize +1
    
    contrastraster = np.zeros((outy,outx), dtype = float)
    contrastraster[:] = 0.0
    
    dissimilarityraster = np.zeros((outy,outx), dtype = float)
    dissimilarityraster[:] = 0.0
    

    homogeneityraster = np.zeros((outy,outx), dtype = float)
    homogeneityraster[:] = 0.0
    

#    energyraster = np.zeros((outy,outx), dtype = float)
#    energyraster[:] = 0.0
        
    correlationraster = np.zeros((outy,outx), dtype = float)
    correlationraster[:] = 0.0
    
    ASMraster = np.zeros((outy,outx), dtype = float)
    ASMraster[:] = 0.0
    
    meanraster = np.zeros((outy,outx), dtype = float)
    meanraster[:]= 0
    
    varraster = np.zeros((outy,outx), dtype = float)
    varraster[:]=0
    
    entroraster = np.zeros((outy,outx), dtype = float)
    entroraster[:]=0
    
#    for jo in tqdm(range(outy)):
    for jo in range(outy):
#    for io in range(outx):
        for io in range(outx):
            i = io + hwin
            j = jo + hwin
            if j >=ny:
                continue
            glcm_win = imgbuf[j-hwin:j+hwin+1,i-hwin:i+hwin+1]
#            try:
            glcm = greycomatrix(glcm_win,[step],[angle],levels=level,symmetric=True,normed= True)
#                glcm = greycomatrix(glcm_win,[step],[angle],levels=level,normed= True)
            correlationraster[jo,io] = greycoprops(glcm,prop = 'correlation')
            contrastraster[jo,io] = greycoprops(glcm,prop = 'contrast')
            dissimilarityraster[jo,io] = greycoprops(glcm,prop = 'dissimilarity')
            homogeneityraster[jo,io] = greycoprops(glcm,prop = 'homogeneity')
            ASMraster[jo,io] = greycoprops(glcm,prop = 'ASM')
#                energyraster[jo,io] = greycoprops(glcm,prop = 'energy')
            entroraster[jo,io] = glcm_other_feas(glcm,prop='entropy')
            meanraster[jo,io] = glcm_other_feas(glcm,prop='mean')
            varraster[jo,io] = glcm_other_feas(glcm,prop='variance')
            glcm = None

    
    return meanraster,varraster,entroraster,contrastraster, dissimilarityraster, homogeneityraster, correlationraster, ASMraster
def saveim(filename,im,srs=None,geo= None):
    if len(im.shape)>2:
        nb,nrow,ncol = im.shape
        driver = gdal.GetDriverByName('GTiff')    
        outDs = driver.Create(filename,ncol,nrow,nb,gdal.GDT_Float32)
        if None is not geo:
            outDs.SetGeoTransform(geo)
        if None is not srs:
            outDs.SetProjection(srs)
        for i in range(nb):
            outBand = outDs.GetRasterBand(i+1)
            outBand.WriteArray(im[i,:,:])
    else:
        nrow,ncol = im.shape
        nb = 1
        driver = gdal.GetDriverByName('GTiff')    
        outDs = driver.Create(filename,ncol,nrow,nb,gdal.GDT_Float32)
        if None is not geo:
            outDs.SetGeoTransform(geo)
        if None is not srs:
            outDs.SetProjection(srs)       
            outBand = outDs.GetRasterBand(nb)
            outBand.WriteArray(im)

    
    outBand = None
    outDs = None

def single_proc_file(imgfile,winsize,ngrey = 256):

    ds = gdal.Open(imgfile)
    nx = ds.RasterXSize
    ny = ds.RasterYSize
##    
    nb = ds.RasterCount
#
    geo = ds.GetGeoTransform()
    srs = ds.GetProjection()
#    dtype = ds.GetRasterBand(1).DataType
    im = ds.ReadAsArray(0,0,nx,ny)
#   
    blue = im[0]
    green = im[1]
    red = im[2]
    nir = im[3]
    
    ndvi = cal_ndvi(red,nir)
    ndvi_dir = './输出数据'
    out_ndvi_path = os.path.join(ndvi_dir ,'ndvi.tif')
    saveim(out_ndvi_path,ndvi,srs=srs,geo=geo)
#    minv,maxv = getminmaxatpercent(im[3,:,:])
#    stretchim = stretch(im[3,:,:],minv,maxv,ngrey,1.8)
#    padded = paddedbuf(stretchim,winsize)
    
#    data = im
    basename = os.path.basename(imgfile).split('.')
#    basedir = os.path.dirname(imgfile)
    basedir = './输出数据'
    for i in range(nb):
        t1 = time.time()
        minv,maxv = getminmaxatpercent(im[i,:,:])
        stretchim = stretch(im[i,:,:],minv,maxv,ngrey,1.8)
        padded = paddedbuf(stretchim,winsize)
        mea,var,entr,con,diss,homo,corr,ASM = glcm_by_win_scan(padded,winsize,step = 1, angle = np.pi/4,level = ngrey)
        
        
#        basename = ['input','tif']
#        basedir = './'
        outname = os.path.join(basedir,basename[0]+'_b{}_win{}_texture.tif'.format(i,winsize))
    
        saveim(outname,np.array([mea,var,entr,con,diss,homo,corr,ASM]),srs=srs,geo=geo)
#        saveim(outname,np.array([mea,var,entr,con,diss,homo,corr,ASM]),srs=None,geo=None)
        t2 = time.time()
        
        print('single process is ok! total time is {},processing file {},band {} \n'.format((t2 -t1)/60.0,basename[0],i))
        
        
if __name__== '__main__':
    imgfile = './使用数据/spot6卫星影像.tif'
    winsize = 7
    single_proc_file(imgfile,winsize,ngrey = 256)
