import os
import csv
from datetime import datetime, timedelta
import pandas as pd
from config import *
from multiprocessing import Process, Queue
import math
inPutPath = SPLIT_DATA_PATH
firstK = 1
def sameDevice(a, b):
    itemList = ['bankgroup','bank']
    for item in itemList:
        if a[item] != b[item]:
            return False
    return True

def adjacentPosition(a, b):
    if (abs(a['column'] - b['column']) <= 30) :
        return True
    return False 
def samePosition(a, b):
    if abs(a['row'] - b['row']) <= 0 and abs(a['column'] - b['column']) <= 0:
        return True
    return False 

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
            
            
            rank,bankgroup,bank,row,column = int(error['rank']), int(error['bankgroup']), int(error['bank']), int(error['row']), int(error['column'])
            bankId = (rank,bankgroup,bank)
            position = (row,column)
            uePosition = (bankId,position)
            
            if uePosition not in UEMap:
                UEMap[uePosition] = errorTime
                UEList.append(error)
                if len(UEList) >= firstK:
                    break
    
    return len(UEList) > 0 , UEMap, UEList

# 统计发生UER的内存条中,首次UER发生前DIMM故障记录
def process(q, dirlist):
    for dir in dirlist:
        df = pd.read_csv(os.path.join(inPutPath,dir,dir+".csv"))
        
        df['record_datetime'] = pd.to_datetime(df['record_datetime'], format="%Y-%m-%d %H:%M:%S")
        
        df = df[(df['with_phy_addr'] == True)].reset_index(drop=True)
        
        UEFLag, UEMap, UEList = getUEMap(df)
        
        
        if not UEFLag:
            q.put([dir,1, [0,0,0,0]])    
        
        unPredictableUERCount , unPredictablePatrolScrubbingUEOCount= 0, 0 
        
        UERList , PatrolScrubbingUEOList = [], []
        for ue in UEList:
            if ue['err_type'] in UETypeList:
                UERList.append([ue, False])
            elif ue['err_type'] in PatrolScrubbingUEOTypeList:
                PatrolScrubbingUEOList.append([ue, False])
        
        df = df[(~df['err_type'].isin(UETypeList))].reset_index(drop=True)

        for i in range(len(UERList)):
            item = UERList[i]
            ue = item[0]
            
            subDf = df[df['record_datetime'] < ue['record_datetime']].reset_index(drop=True)
            if subDf.shape[0] == 0:
                unPredictableUERCount += 1
                continue
            for _, error in  subDf.iterrows():
                if ue['record_datetime'] - error['record_datetime'] > timedelta(minutes=5) and sameDevice(error, ue) and adjacentPosition(error, ue):
                    UERList[i][1] = True
                    
        for i in range(len(PatrolScrubbingUEOList)):
            item = PatrolScrubbingUEOList[i]
            ue = item[0]
            
            subDf = df[df['record_datetime'] < ue['record_datetime']].reset_index(drop=True)
            if subDf.shape[0] == 0:
                unPredictablePatrolScrubbingUEOCount += 1
                continue
            for _, error in  subDf.iterrows():
                if ue['record_datetime'] - error['record_datetime'] > timedelta(minutes=5) and sameDevice(error, ue) and adjacentPosition(error, ue):
                    PatrolScrubbingUEOList[i][1] = True

        uerCount = len(UERList) - unPredictableUERCount
        predictableUERCount = 0
        for item in UERList:
            predictableUERCount += item[1] == True
        
        PatrolScrubbingUEOCount = len(PatrolScrubbingUEOList) - unPredictablePatrolScrubbingUEOCount
        predictablePatrolScrubbingUEOCount = 0
        for item in PatrolScrubbingUEOList:
            predictablePatrolScrubbingUEOCount += item[1] == True
        
        q.put([dir,2, [uerCount, predictableUERCount, PatrolScrubbingUEOCount, predictablePatrolScrubbingUEOCount]])    
        
        

def merge(q):
    count = [0,0]
    strSet = set()
    uerCount, predictableUERCount, PatrolScrubbingUEOCount, predictablePatrolScrubbingUEOCount = 0,0,0,0
    while True:
        str, op, countList = q.get()
        if op == 0:
            print(uerCount, predictableUERCount, PatrolScrubbingUEOCount, predictablePatrolScrubbingUEOCount)
            return
        if op > 1:
            uerCount += countList[0]
            predictableUERCount += countList[1]
            PatrolScrubbingUEOCount += countList[2]
            predictablePatrolScrubbingUEOCount += countList[3]
         
def main():
    dirlist = os.listdir(inPutPath)
    processList = []
    cpuCount = os.cpu_count() * 2
    subListSize = math.ceil(len(dirlist) / cpuCount)
    q = Queue()
    
    for i in range(cpuCount):
        subDimm = dirlist[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=process, args=([q, subDimm])))
    
    pMerge = Process(target=merge, args=([q]))
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    q.put([dir,0, [0,0,0,0]]) 
    pMerge.join()
                
main()
