# -*- coding: UTF-8 -*-
# 创建不同长宽的数据集，存放在new_datase2 和 new_dataset3 中
import os
import csv
from datetime import datetime, timedelta
import shutil
import numpy as np 
import math
from multiprocessing import Process, Queue
from config import *
import pandas as pd
import pickle
from tqdm import tqdm
# split_by_DIMMSN.py 和 split_by_bank.py 处理得到数据的路径

inPutPath = SPLIT_DATA_PATH
MAX_COLUMN = 10
MAX_ROW = 10
subBankLifeSpan = timedelta(minutes=5)
# 存储数据集的路径
data_savepath = os.path.join(DATA_SET_PATH, "transformer")
firstK = 1
datasetName = 'first{}_{}_{}.pickle'.format(firstK, MAX_ROW*2, MAX_COLUMN*2)

UER_INTERVAL = timedelta(days=1)

def merge(q, x):

    count ,count1, = 0, 0
    dataSet = {'positive':{}, 'negative':{}}

    
    bar = tqdm(range(x))
    for i in bar:
        dir,sampleList,UEFlag, predictFlag = q.get()
        if len(sampleList) == 0:
            continue

        if UEFlag > 0:
            count += UEFlag
            if predictFlag :
                count1 += predictFlag
        if predictFlag > 0:
            dataSet['positive'][dir] = [sampleList]
        else:
            dataSet['negative'][dir] = [sampleList]

        
    print(count, count1)
    file = datasetName
    with open(file, 'wb') as f:
        pickle.dump(dataSet, f)
    
        
        

# 分析当前 error 是否在 window 中
def getPosition(errposition, subBankCenter, maxRow ,maxColumn):
    distance = [abs(errposition[0] - subBankCenter[0]), abs(errposition[1] - subBankCenter[1])]
    # 若 err 位置在 window 中 
    rd = [int(errposition[0] - subBankCenter[0] + maxRow), int(errposition[1] - subBankCenter[1] + maxColumn)]
    if distance[0] >= maxRow and distance[1] >= maxColumn:
        return False, [-1,-1]
    
    if distance[0] < maxRow and distance[1] < maxColumn:
        return True, rd
            
    elif distance[0] < maxRow or distance[1] < maxColumn:
        rd[0] = max(rd[0], 0)
        rd[0] = min(rd[0], 2*maxRow - 1)
        rd[1] = max(rd[1], 0)
        rd[1] = min(rd[1], 2*maxColumn - 1)
    rd = [int(rd[0]), int(rd[1])]
    return True, rd




def addCenter(subBankList, error, maxRow, maxColumn,UEMap):
    rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
    bankId = (rank,bankgroup,bank)
    position = (row,column)
    errorTime = error['record_datetime']
    

    
    for subBank in subBankList:
        subBankBankId = subBank[0]
        subBankEndTime = subBank[1]
        subBankCenter = subBank[2]
        if (subBankBankId == bankId and 
            errorTime <= subBankEndTime and 
            (abs(subBankCenter[0] - position[0]) < maxRow and abs(subBankCenter[1] - position[1]) < maxColumn)):
            return subBankList
    subBankList.append((bankId, errorTime + subBankLifeSpan, position))
            
    return subBankList

def getUEMap(df):
    UEMap= {}
    UEList = []
    UERDf = df[(df['err_type'].isin(UETypeList))].reset_index(drop=True)
    if UERDf.shape[0] != 0:
        for _, error in UERDf.iterrows():
            
            errorTime = error['record_datetime']
            flag = False
            for ue in UEList:
                if errorTime - ue['record_datetime'] < timedelta(minutes=5):
                    flag = True
            if flag:
                continue
            
            
            rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
            bankId = (rank,bankgroup,bank)
            position = (row,column)
            uePosition = (bankId,position)
            
            if uePosition not in UEMap:
                UEMap[uePosition] = errorTime
                UEList.append(error)
                if len(UEList) >= firstK:
                    break
    
    return len(UEList) > 0 , UEMap, UEList
# 对于连续一段时间内发生的UE，仅有第一个造成致命效果
# 一个UE发生后，之后一段时间的UE都视为无影响

# 剩余的致命UEs，我们只取前若干个，etc 10个，假设出现10次->换条，踢出数据集

def observeUE(UEMap, bankId, subBankCenter ,subBankEndTime):
    
    UESet = set()
    
    UEPositionList =list(UEMap.keys())
    for i in range (len(UEPositionList)):
        ue = UEPositionList[i]
        UEBankId = ue[0]
        uePosition = ue[1]
        ueTime = UEMap[ue]
        if (UEBankId == bankId 
            and ueTime > subBankEndTime 
            and (abs(subBankCenter[0] - uePosition[0]) < MAX_ROW 
                #  and abs(subBankCenter[1] - uePosition[1]) < MAX_COLUMN
                 )
            ):
            UESet.add(i)

    return len(UESet) > 0, UESet
def process(q, dirlist, interval, maxRow, maxColumn):
    for dir in dirlist:
        # print(dir)
        file = os.path.join(inPutPath,dir,dir+".csv")
        df = pd.read_csv(file, low_memory=False)
        df['record_datetime'] = pd.to_datetime(df['record_datetime'], format="%Y-%m-%d %H:%M:%S")
        df = df[(df['with_phy_addr'] == True) ].reset_index(drop=True)
        
        UEFlag , UEMap, UEList = getUEMap(df)
        if UEFlag:
            lastUER = UEList[len(UEList)-1]['record_datetime']
            df = df[df['record_datetime'] < lastUER].reset_index(drop=True)
        
        visibleUESet = set()
        
        CEDf = df[(~df['err_type'].isin(UETypeList))].reset_index(drop=True)
        # CEDf = df[ (df['err_type'] == 'CE')].reset_index(drop=True)

        subBankList = []
        errorList = []
        for _ , ce in CEDf.iterrows():
            errorList.append(ce)
            # if ce['err_type'] not in UETypeList:
            subBankList = addCenter(subBankList, ce, maxRow, maxColumn,UEMap)
        
        errorList += UEList
        sampleList = []
        for subBank in subBankList:
            
            subBankBankId = subBank[0]
            subBankEndTime = subBank[1]
            subBankCenter = subBank[2]
            sample = [np.zeros((int(maxRow*2),int(maxColumn*2),3),dtype = np.uint8), False, subBankEndTime, dir, set(),subBankCenter]
            
            # y
            # for error in errorList:
            #     rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
            #     bankId = (rank,bankgroup,bank)
            #     position = (row,column)
                
            #     errorTime = error['record_datetime']
            #     errorType = error['err_type']
            #     if (subBankBankId == bankId and 
            #         errorTime > subBankEndTime):
            #         flag, positionInSubBank = getPosition(position, subBankCenter, maxRow, maxColumn)

            #         if flag:
            #             if errorType == 'CE':
            #                 sample[0][positionInSubBank[0]][positionInSubBank[1]][0] += 1
            #             elif errorType == 'PatrolScrubbingUEO':
            #                 sample[0][positionInSubBank[0]][positionInSubBank[1]][1] += 1
            #             elif errorType in UETypeList:
            #                 sample[0][positionInSubBank[0]][positionInSubBank[1]][2] += 1
            #             sample[2] = errorTime
            
            
            observeUEFlag,UESet = observeUE(UEMap, subBankBankId,subBankCenter,subBankEndTime)
            if observeUEFlag :
                sample[1] = True
                sample[4] = UESet
                for ue in UESet:
                    visibleUESet.add(ue)

            sampleList.append(sample)

        q.put([dir,sampleList,len(UEMap), len(visibleUESet)])
        
def main(interval, maxRow, maxColumn):
    processList = []
    cpuCount = os.cpu_count() * 3
    # cpuCount = 1
    
    dirlist = os.listdir(inPutPath)
    # dirlist = dirlist[:10]
    subListSize = math.ceil(len(dirlist) / cpuCount)
    q = Queue()
    for i in range(cpuCount):
        subDimm = dirlist[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=process, args=([q, subDimm,interval, maxRow, maxColumn])))
    pMerge = Process(target=merge, args=([q, len(dirlist)]))
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    # q.put(["",'','',''])
    pMerge.join() 
main(24,MAX_ROW,MAX_COLUMN)
