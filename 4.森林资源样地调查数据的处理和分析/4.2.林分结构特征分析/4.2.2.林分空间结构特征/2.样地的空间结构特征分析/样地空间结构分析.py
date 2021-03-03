# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

def get_distance(file, inX):
#    inX = 0
    dataSet = pd.read_csv(file, encoding="utf_8_sig",engine='python',keep_default_na = False)
    tree_points = dataSet[['x', 'y']]#获取样地所有点坐标的列表
    tree_points = tree_points.values#取坐标列表中的值

    dataSetSize = dataSet.shape[0]#获取数据行数
    tree = tree_points[inX]#随机当前树
    diffMat = np.tile(tree, (dataSetSize, 1)) - tree_points#使当前树与其他树做减法
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    return [sortedDistIndicies[0:5], distances[sortedDistIndicies[0:5]]]

def get_vector_included_angle2(tree_points,ids):
    central_tree_id = ids[0][0]
    last_four_tree_ids = ids[0][1:]

    degrees = []
    
    for id in last_four_tree_ids[0:]:
        first_vec = tree_points[id] - tree_points[central_tree_id]
        x = first_vec[0]
        y = first_vec[1]
        
        theta = np.math.atan2(y,x)*180.0/np.pi
        if theta <0:
            theta = 360+theta
        degrees.append(theta)
    degrees.sort()
    angles = [degrees[1]- degrees[0],degrees[2]-degrees[1],degrees[3]-degrees[2],degrees[3]-degrees[0]]
    for i,_ in enumerate(angles):
        if angles[i]>180:
            angles[i] = 360-angles[i]
    
    return angles
        
def comparsion(degree_list):#角尺度

    a = []
    for i in degree_list:
        if i < 72:
            Z = 1
        else:
            Z = 0
        a.append(Z)
    W = np.sum(a) / 4
    return W

def comparison_DBH(dbhs):#大小比
    b = []
    dia_list = list(dbhs)
    refrence_dia = dia_list[0]
    for d in dia_list[1:]:
        if d < refrence_dia:
            k = 0
        else:
            k = 1
        b.append(k)

    M = np.sum(b) / 4
    return M

def get_average_diameter(diameter):
    D_list = []
    for i in diameter:
        d = i**2
        D_list.append(d)
    D = (np.sum(D_list) / len(diameter))**0.5
    return D

def get_each_columns_average(dataSet):
    dataSet = pd.DataFrame(dataSet)
    list_average = []
    for col in dataSet.columns[5:]:
        average = dataSet['%s'%col].mean()
        list_average.append(average)
    dic = dict(zip(dataSet.columns[5:],list_average))

    return dic

def mingling(class_list):#混交度
    lst = []
    first = list(class_list)[0]
    for i in list(class_list)[1:]:
        if i == first:
            k = 0
        else:
            k = 1
        lst.append(k)
    h = np.sum(lst)/4
    return h
    
    
if __name__ == '__main__':
    file_path = r'./使用数据'       
    out_path  = r'./输出数据'
    
    name_list = os.listdir(file_path)    
    #name_list.sort(key = lambda i:int(re.search(r'(\d+)',i).group()))  
    for j in name_list:
        file = os.path.join(file_path,j)
        dataSet = pd.read_csv(file, encoding="utf_8_sig",engine='python')
        tree_points = dataSet[['x', 'y']]
        tree_points = tree_points.values
        dataSetSize = dataSet.shape[0]
        sps = dataSet[u'树种']
        diameter = dataSet[u'胸径']
        D_average = get_average_diameter(diameter)
        
        idx = []
        tree_min_ids = []
        W_list = []
        M_list = []
        H_list = []
        for i in range(len(tree_points)):
    #        if (i == 6):
    #            print('test')
            ids = get_distance(file,i)
            idx.append(ids)
            
            angles = get_vector_included_angle2(tree_points,idx[i])
            H = mingling(sps[ids[0]])
            H_list.append(H)
            W = comparsion(angles)
            W_list.append(W)
            U = comparison_DBH(diameter[ids[0]])
            M_list.append(U)
        mean = get_each_columns_average(dataSet)
    
        out_df = pd.DataFrame()
        out_df['树种'] = sps
        out_df['角尺度'] = W_list
        out_df['大小比'] = M_list
        out_df['混交度'] = H_list
        out_df.to_csv(os.path.join(out_path,'{}空间结构.csv'.format(j[:-4])),encoding = 'utf_8_sig',index = False)
