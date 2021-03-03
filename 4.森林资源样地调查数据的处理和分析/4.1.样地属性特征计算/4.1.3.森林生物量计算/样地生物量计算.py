# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:28:39 2020

@author: 15771
"""
import pandas as pd 
import numpy as np 
import math

def maweisong(*args): #马尾松及其他松类
    w = 0.0292*(D**2.8301)+0.0021*(D**3.2818)+0.0021*(D**2.8392)+0.0194*(D**2.3497)
    return w
def lilei(*args):#栎类
    w = 0.1030*(D**2.2950)+0.0160*(D**2.6080)+0.0110*(D**2.2170)+0.1280*(D**2.201)
    return w 
def shamu(*args):#杉木及其他杉类
    w = 0.0543*(D**2.4242)+0.0255*(D**2.0726)+0.0773*(D**1.5761)+0.0513*(D**2.0338)
    return w 
def baimu(*args):#柏木
    w = 0.0937*(D**2.2225)+0.0323*(D**2.3338)+0.0236*(D**2.3106)+0.057*(D**2.1651)
    return w
def yunshan(*args):#云杉、冷杉
    w = 0.0562*(D**2.4608)+0.1298*(D**1.8070)+0.1436*(D**1.6729)+0.0313*(D**2.3049)
    return w
def huamu(*args):#桦木
    w = 0.1555*(D**2.2273)+0.0134*(D**2.4932)+0.0092*(D**2.0967)+0.0242*(D**2.4750)
    return w
def luoyesong(*args):#落叶松
    w = 0.0526*(D**2.5257)+0.0085*(D**2.4815)+0.0168*(D**2.0026)+0.0291*(D**2.2645)
    return w
def hongsong(*args):#红松
    w = 0.1087*(D**2.1527)+0.0481*(D**2.0877)+0.0631*(D**1.8343)+0.0305*(D**2.3298)
    return w
def yunnansong(*args):#云南松
    w = 0.0900*(D**3.4678)+0.0310*(D**3.3250)+0.0298*(D**2.3596)+0.4432*(D**2.6927)
    return w
def huashansong(*args):#华山松
    w = 0.0787*(D**2.2823)+0.0270*(D**2.3664)+0.0046*(D**2.5540)+0.0224*(D**2.2836)
    return w
def yousong(*args):#油松
    w = 0.1450*(D**2.1567)+0.0673*(D**1.9781)+0.0600*(D**1.9329)+0.0503*(D**2.0886)
    return w
def zhangzisong(*args):#樟子松
    w = 0.0840*(D**2.2337)+0.0691*(D**1.7370)+0.0994*(D**1.8157)+0.2645*(D**1.4197)
    return w
def qitayingkuo(*args):#其他硬阔类
    w = 0.0971*(D**2.3253)+0.0278*(D**2.3540)+0.0239*(D**2.0051)+0.0529*(D**2.2317)
    return w
def anshu(*args):#桉树
    w = 0.0349*(D**2.7913)+0.0701*(D**1.7318)+0.0175*(D**2.4165)+0.0186*(D**2.3163)
    return w
def yangshu(*args):#杨树
    w = 0.0800*(D**2.3350)+0.0210*(D**2.3400)+0.0120*(D**2.0130)+0.0360*(D**2.1920)
    return w
def qitaruankuo(*args):#其他软阔类
    w = 0.1286*(D**2.2255)+0.0445*(D**1.9516)+0.0197*(D**1.6667)+0.0630*(D**2.0316)
    return w
def tieshan(*args):#铁杉、柳杉、油杉
    w = 0.1909*(D**1.9859)+0.0205*(D**2.2230)+0.0453*(D**1.8432)+0.0223*(D**2.3840)
    return w
def zhenyelin(*args):#针叶林
    w = 0.0670*(D**2.4090)+0.0220*(D**2.2700)+0.0250*(D**2.1240)+0.0380*(D**2.1650)
    return w
def kuoyelin(*args):#阔叶混
    w = 0.1300*(D**2.2010)+0.0380*(D**2.1650)+0.0130*(D**2.0630)+0.0570*(D**2.1710)
    return w
def zhenkuohunjiaolin(*args):#针阔混
    w = 0.0610*(D**2.4590)+0.0970*(D**1.8460)+0.1330*(D**1.4550)+0.0960*(D**1.9280)
    return w
def shuanpili(*args):#栓皮栎
    w = 0.0554*(D**1.7229)+0.0141*(D**1.5988)+0.0120*(D**1.6030)+0.0261*(D**1.6906)
    return w
def xiangzhang(*args):#香樟
    w = 0.0250*(D**2.7300)+0.0020*(D**3.0880)+0.0070*(D**2.4470)+0.0070*(D**2.8190)
    return w
def muhe(*args):#木荷
    w = 0.0803*(D**2.4096)+0.1857*(D**1.6599)+0.0246*(D**1.8790)+0.0343*(D**2.2451)
    return w
def fengxiang(*args):#枫香
    w = 0.1997*(D**2.0511)+0.1537*(D**1.6627)+1.3336*(D**0.5549)+0.2450*(D**1.6905)
    return w
def shuishan(*args):#水杉
    w = 0.0287*(D**2.6604)+0.0402*(D**1.9898)+0.1317*(D**1.2972)+0.0172*(D**2.3939)
    return w
def qinggangli(*args):#青冈栎
    w = 0.1366*(D**2.2845)+0.0069*(D**2.9094)+0.0011*(D**3.0210)+0.0434*(D**2.4476)
    return w
if __name__ =='__main__':
    file = open(r'./使用数据/2014交集.csv')
    data = pd.read_csv(file)
    df = pd.DataFrame(data)
    name_list = list(df['树种名称'])
    new_list = []
    for i in range(len(name_list)):
        if name_list[i] == '马尾松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)
        elif name_list[i] == '栓皮栎':#栓皮栎
            D = df.loc[i]['胸径']
            shuanpili_val = shuanpili(D)
            new_list.append(shuanpili_val)
        elif name_list[i] == '栎':
            D = df.loc[i]['胸径']
            lilei_val = lilei(D)
            new_list.append(lilei_val)
        elif name_list[i] == '其他软阔类':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)    
        elif name_list[i] == '柏木':#柏木
            D = df.loc[i]['胸径']
            baimu_val = baimu(D)
            new_list.append(baimu_val)
        elif name_list[i] == '榆':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)
        elif name_list[i] == '樟':#香樟
            D = df.loc[i]['胸径']
            xiangzhang_val = xiangzhang(D)
            new_list.append(xiangzhang_val)
        elif name_list[i] == '其他硬阔':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)
        elif name_list[i] == '杨树':
            D = df.loc[i]['胸径']
            yangshu_val = yangshu(D)
            new_list.append(yangshu_val)
        elif name_list[i] == '杉木':#杉木及其他杉类
            D = df.loc[i]['胸径']
            shamu_val = shamu(D)
            new_list.append(shamu_val)    
        elif name_list[i] == '柳树':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)    
        elif name_list[i] == '黑松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)    
        elif name_list[i] == '泡桐':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)
        elif name_list[i] == '楝树':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)
        elif name_list[i] == '兴山榆（天然）':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)
        elif name_list[i] == '桦木':
            D = df.loc[i]['胸径']
            huamu_val = huamu(D)
            new_list.append(huamu_val)
        elif name_list[i] == '白桦':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)    
        elif name_list[i] == '椴树':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)
        elif name_list[i] == '铁杉':
            D = df.loc[i]['胸径']
            tieshan_val = tieshan(D)
            new_list.append(tieshan_val)
        elif name_list[i] == '柳杉':
            D = df.loc[i]['胸径']
            tieshan_val = tieshan(D)
            new_list.append(tieshan_val)
        elif name_list[i] == '枫香':
            D = df.loc[i]['胸径']
            fengxiang_val = fengxiang(D)
            new_list.append(fengxiang_val)
        elif name_list[i] == '湿地松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)   
        elif name_list[i] == '油松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            yousong_val = yousong(D)
            new_list.append(yousong_val)  
        elif name_list[i] == '其他杉类':#杉木及其他杉类
            D = df.loc[i]['胸径']
            shamu_val = shamu(D)
            new_list.append(shamu_val)
        elif name_list[i] == '油杉':
            D = df.loc[i]['胸径']
            tieshan_val = tieshan(D)
            new_list.append(tieshan_val)
        elif name_list[i] == '华山松':
            D = df.loc[i]['胸径']
            huashansong_val = huashansong(D)
            new_list.append(huashansong_val)
        elif name_list[i] == '日本落叶松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)   
        elif name_list[i] == '楠':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)    
        elif name_list[i] == '冷杉':
            D = df.loc[i]['胸径']
            yunshan_val = yunshan(D)
            new_list.append(yunshan_val)
        elif name_list[i] == '其他松类': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)   
        elif name_list[i] == '池杉':#杉木及其他杉类
            D = df.loc[i]['胸径']
            shamu_val = shamu(D)
            new_list.append(shamu_val)   
        elif name_list[i] == '水杉':
            D = df.loc[i]['胸径']
            shuishan_val = shuishan(D)
            new_list.append(shuishan_val)    
        elif name_list[i] == '黄山松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)       
        elif name_list[i] == '檫木':#其他软阔类
            D = df.loc[i]['胸径']
            qitaruankuo_val = qitaruankuo(D)
            new_list.append(qitaruankuo_val)   
        elif name_list[i] == '乌桕':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '柿':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '油桐':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '其他（药材）':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '杏':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '蚕桑':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '核桃':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '枣':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '桃':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '白蜡树':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '其他（果树）':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '山楂':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '其他（药材）漆树':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '板栗':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '梨':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '李':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '油茶':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '杜仲':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '其他（林化业原料）':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '肉桂':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '银杏':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '黄柏':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '八角':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '杜鹃':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '蚕柞':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '苹果':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '厚朴':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '其他（食用原料）':#阔叶林
            D = df.loc[i]['胸径']
            kuoyelin_val = kuoyelin(D)
            new_list.append(kuoyelin_val)
        elif name_list[i] == '枫桦':#其他硬阔类
            D = df.loc[i]['胸径']
            qitayingkuo_val = qitayingkuo(D)
            new_list.append(qitayingkuo_val)  
        elif name_list[i] == '木荷':#其他硬阔类
            D = df.loc[i]['胸径']
            muhe_val = muhe(D)
            new_list.append(muhe_val)       
        elif name_list[i] == '火炬松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)
        elif name_list[i] == '红豆杉（天然）': #马尾松及其他松类
            D = df.loc[i]['胸径']
            zhenyelin_val = zhenyelin(D)
            new_list.append(zhenyelin_val)
        elif name_list[i] == '国外松': #马尾松及其他松类
            D = df.loc[i]['胸径']
            maweisong_val = maweisong(D)
            new_list.append(maweisong_val)  
    df['bio'] = new_list
    #print(df)
    df.to_csv(r'./输出数据\2014样木生物量1111.csv',index = False, encoding="gbk")