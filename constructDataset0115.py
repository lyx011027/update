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
    length = [0,0]
    for i in dataSet['positive'].values():
        # print(i)
        length[0] += len(i)
    for i in dataSet['negative'].values():
        length[1] += len(i)
    print(length)
    file = datasetName
    with open(file, 'wb') as f:
        pickle.dump(dataSet, f)
    
        


def getUEMap(df):
    UEMap= {}
    UEList = []
    UERDf = df[(df['err_type'].isin(UETypeList))].reset_index(drop=True)

    for _, error in UERDf.iterrows():
        
        errorTime = error['record_datetime']

        
        rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
        bankId = (rank,bankgroup,bank)
        position = (row,column)
        uePosition = (bankId,position)
        
        if uePosition not in UEMap:
            UEMap[uePosition] = errorTime
            UEList.append(error)

    return len(UEList) > 0 , UEMap, UEList


def getUERowMap(UEList):
    UERowMap = {}
    for error in UEList:
        rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
        rowId = (rank,bankgroup,bank,row)
        if rowId not in UERowMap:
            
            UERowMap[rowId] = error['record_datetime']
    return UERowMap
def getAdjacentErrorList(cellMap, currnetCellId, maxRow):
    errorList = []
    columnSet ,rowSet= set(), set()
    for cellId in cellMap.keys():
        if cellId[:3] == currnetCellId[:3] and ((abs(cellId[3] - currnetCellId[3]) <= maxRow)) :
            errorList.append(cellId)
            columnSet.add(cellId[4])
            rowSet.add(cellId[3])
            
    if len(rowSet) >= 2:
        return errorList
    return []

regionWidth = int(1024 /8 / 4)

def process(q, dirlist, interval, maxRow, maxColumn):
    for dir in dirlist:
        # print(dir)
        file = os.path.join(inPutPath,dir,dir+".csv")
        df = pd.read_csv(file, low_memory=False)
        df['record_datetime'] = pd.to_datetime(df['record_datetime'], format="%Y-%m-%d %H:%M:%S")
        df = df[(df['with_phy_addr'] == True) ].reset_index(drop=True)
        df['column'] = df['column']
        observeFlag = False
        
        UEFlag , UEMap, UEList = getUEMap(df)
        ueRowMap = getUERowMap(UEList)
        
        CEDf = df[(df['err_type'] == 'CE')].reset_index(drop=True)
        sampleList = []
        cellMap = {}
        for _, error in CEDf.iterrows():
            rank,bankgroup,bank,row,column =  int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
            bankId = (rank,bankgroup,bank)
            cellId = (rank,bankgroup,bank,row,column)
            position = (row,column)
            errorTime = error['record_datetime']
            
            if cellId in cellMap:
                cellMap[cellId] += 1
                continue
            # else:
            cellMap[cellId] = 1
            
            regionCenter = (row, int(column/regionWidth)*regionWidth)
            # if column > 32:
            #     print(regionCenter)
            
            '''
            
             0: 21*32 矩阵
             1: row 是否有UE [bool]*21
             2: 样本生成时间
             3: dimm_sn
             4: region 中心位置[rowid, columnId] , i行预测为故障, region[0] + i - 10 
             5: 相邻行中是否存在UE
            '''
            sample = [np.zeros((int(maxRow*2 + 1),regionWidth),dtype = np.uint8), np.zeros(int(maxRow*2 + 1), dtype=bool), errorTime, dir,regionCenter,False]
            
            adjacentErrorList = getAdjacentErrorList(cellMap, cellId, maxRow)
            
            if len(adjacentErrorList) == 0:
                continue
            # maxrow = 0, maxrow*2 = maxrow
            for adjacentError in adjacentErrorList:
                rd = [int(adjacentError[3] - regionCenter[0] + maxRow), int(adjacentError[4] - regionCenter[1])]
                rd[0] = max(rd[0], 1)
                rd[0] = min(rd[0], 2*maxRow - 1)
                rd[1] = max(rd[1], 0)
                rd[1] = min(rd[1], regionWidth - 1)   
                # print(adjacentError[4] , regionCenter[1])
                sample[0][rd[0]][rd[1]] = cellMap[adjacentError]
                # print(1)
            
            for adjacentRow in range(cellId[3]-maxRow + 1, cellId[3] + maxRow):
                rowId = cellId[:3] + (adjacentRow,)
                if rowId in ueRowMap and errorTime < ueRowMap[rowId] and   ueRowMap[rowId] - errorTime <timedelta(days=1):
                    sample[1][int(adjacentRow - cellId[3] + maxRow)] = True
                    observeFlag = True
                    sample[5] = True
                    # print(adjacentRow - cellId[3] + maxRow)
                    # print(dir)
            
            
            
            
  

            sampleList.append(sample)
        
        q.put([dir,sampleList,UEFlag,observeFlag])
        
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
