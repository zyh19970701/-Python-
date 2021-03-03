# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:41:28 2021

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
import os
from scipy.optimize import curve_fit
import numpy as np
import math
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

def func(x, a,u, sig):
    return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))

def get_curve_fit_param(x,y,opt_fun,p0):    
    poptg,pcov = curve_fit(opt_fun,x,y,p0=p0,maxfev = 60000) #bounds=(0, [2, 24., 2])
    err = np.sqrt(np.sum((opt_fun(x,*poptg) - y)**2)/len(x))
    print(err)
    print(poptg)
    print(pcov)
    return poptg

def weib(x,a, scale, shape):
    return 44*(shape / scale) * ((x-a) / scale)**(shape - 1) * np.exp(-((x-a) / scale) ** shape)
def weibp(x,a, scale, shape):
    return (shape / scale) * ((x-a) / scale)**(shape - 1) * np.exp(-((x-a) / scale) ** shape)

dir_path = r'./使用数据'
DFs = []
for root, dirs, files in os.walk(dir_path):  #第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    for file in files:
        file_path=os.path.join(root,file)  #将路径名和文件名组合成一个完整路径
        df = pd.read_excel(file_path) #excel转换成DataFrame
        DFs.append(df)
        
df= pd.concat(DFs)
df = df.query('50>胸径>20')

"""
胸径树高频率频数分布拟合
"""
x=np.array(df['胸径'])
fig = plt.figure(figsize=(10, 6))
ax = Subplot(fig, 221)
fig.add_subplot(ax)
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)
n,bins,c = ax.hist(df['胸径'],bins=range(23,51,2),edgecolor='black')
y=n
x=bins[:-1]
ztfb = get_curve_fit_param(x,y,func,p0=[5,35,6])    
ztfbpre = func(x,*ztfb)
# ax.plot(x,y,'r*',ls='-')
ax.plot(x,ztfbpre,'b+',ls='-',color='orange')
wbfb = get_curve_fit_param(x,y,weib,p0=[20,20,5])    
wbfbpre = weib(x,*wbfb)
# plt.plot(x,y,'r*')
ax.plot(x,wbfbpre,'b+',ls='-')
plt.xticks(range(24,51,2))
plt.xlabel('径阶分布（cm）')
plt.ylabel('频数')
plt.yticks([2,4,6,8])
ax1 = Subplot(fig, 222)
fig.add_subplot(ax1)
ax1.axis["right"].set_visible(False)
ax1.axis["top"].set_visible(False)
n1,bins1,c1 = ax1.hist(df['树高'],bins=range(11,23,1),edgecolor='black')
y1=n1
x1=bins1[:-1]
ztfb1 = get_curve_fit_param(x1,y1,func,p0=[2,19,6])    
ztfbpre1 = func(x1,*ztfb1)
# ax.plot(x,y,'r*',ls='-')
ax1.plot(x1,ztfbpre1,'b+',ls='-',color='orange')
wbfb1 = get_curve_fit_param(x1,y1,weib,p0=[10,20,5])    
wbfbpre1 = weib(x1,*wbfb1)
# plt.plot(x,y,'r*')
ax1.plot(x1,wbfbpre1,'b+',ls='-')
plt.xticks(range(11,23,1))
plt.xlabel('树高阶分布（m）')
plt.ylabel('频数')

ax2 = Subplot(fig, 223)
fig.add_subplot(ax2)
ax2.axis["right"].set_visible(False)
ax2.axis["top"].set_visible(False)
n2,bins2,c2 = ax2.hist(df['胸径'],bins=range(23,51,2),edgecolor='black',density=True)
y2=n2/np.sum(n2)
x2=bins2[:-1]
ztfb2 = get_curve_fit_param(x2,y2,func,p0=[5,35,6])    
ztfbpre2 = func(x2,*ztfb2)
# ax.plot(x,y,'r*',ls='-')
ax2.plot(x2,ztfbpre2,'b+',ls='-',color='orange')
wbfb2 = get_curve_fit_param(x2,y2,weibp,p0=[15,20,5])    
wbfbpre2 = weibp(x2,*wbfb2)
# plt.plot(x,y,'r*')
ax2.plot(x2,wbfbpre2,'b+',ls='-')
plt.xticks(range(24,51,2))
plt.xlabel('径阶分布（cm）')
plt.ylabel('频率')
ax3 = Subplot(fig, 224)
fig.add_subplot(ax3)
ax3.axis["right"].set_visible(False)
ax3.axis["top"].set_visible(False)
n3,bins3,c3 = ax3.hist(df['树高'],bins=range(11,23,1),edgecolor='black',density=True)
y3=n3/np.sum(n3)
x3=bins3[:-1]
ztfb3 = get_curve_fit_param(x3,y3,func,p0=[5,35,6])    
ztfbpre3 = func(x3,*ztfb3)
# ax.plot(x,y,'r*',ls='-')
ax3.plot(x3,ztfbpre3,'b+',ls='-',color='orange')
wbfb3 = get_curve_fit_param(x3,y3,weibp,p0=[10,20,5])    
wbfbpre3 = weibp(x3,*wbfb3)
# plt.plot(x,y,'r*')
ax3.plot(x3,wbfbpre3,'b+',ls='-')
plt.xticks(range(10,23,1))
plt.xlabel('树高阶分布（m）')
plt.ylabel('频率')
plt.savefig(r'./输出数据/频数频率分布.jpg',dpi=300)
plt.show()        

"""
3d胸径树高频数频率分布
"""
fig = plt.figure(dpi=500)
ax = fig.add_subplot(121, projection='3d')
x = df['胸径']
y = df['树高']
hist, xedges, yedges = np.histogram2d(x, y)
# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
# Construct arrays with the dimensions for the 16 bars.
dx = dy = 1 * np.ones_like(zpos)
dz = hist.ravel()
dz1 = dz/np.sum(dz)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',edgecolor='black')
ax.tick_params(labelsize=4)
fontdict={'fontsize':5}
ax.set_xlabel('D(cm)',fontdict)
ax.set_ylabel('H(m)',fontdict)
ax.set_zlabel('频数',fontdict)
plt.xticks(range(24,51,2))
plt.yticks(range(10,23,1))
ax1= fig.add_subplot(122, projection='3d')
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, zsort='average',edgecolor='black')
ax1.tick_params(labelsize=4)
fontdict={'fontsize':5}
ax1.set_xlabel('D(cm)',fontdict)
ax1.set_ylabel('H(m)',fontdict)
ax1.set_zlabel('频率',fontdict)
plt.xticks(range(24,51,2))
plt.yticks(range(10,23,1))
ax.view_init(45, 60)
ax1.view_init(45, 60)
plt.savefig(r'./输出数据/胸径树高频数频率分布.jpg',dpi=300)
plt.show()
        
        
        