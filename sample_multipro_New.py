import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *
import os
import math
import copy
from multiprocessing import Process, Queue
# import moxing as mox
MAXROW = 20*8
MAXCOLUMN = 30


bitFlag = False
capacityFlag = False
bit_width_x = 4
capacity = 32*1024

staticItem = [
    'dimm_sn'
    ] + STATIC_ITEM



sampleItem = staticItem + dynamicItem  + ['time','label']



if not os.path.exists(DATA_SET_PATH):
    os.makedirs(DATA_SET_PATH)


def get_writer(dataset):
    f1 = open(dataset, mode="w")
    writer = csv.DictWriter(f1, sampleItem)
    itemMap = {}
    for item in sampleItem:
        itemMap [item] = item
    writer.writerow(itemMap)
    return writer

# 通过静态信息生成basesample
def getBaseSample(dimm, staticFile):
    staticDf = pd.read_csv(staticFile)
    sample = getDynamicSample()
    sample['dimm_sn'] = dimm
    
    return sample


def parseBitError(parity):
    
    if pd.isna(parity):
        return [-1] * 7
    
    adjDqCount = 0
    DQMap = {}
    BurstList = []
    
    
    for i in range(8):
        DQ = int(parity[i + 2], 16)
        for j in range(3):
            if (DQ >>j & 3) == 3:
                adjDqCount += 1
                
        for j in range(4):
            if (DQ >>j & 1) == 1:
                DQMap[4-j-1] = True
        if DQ > 0:
            BurstList.append(i)
    DQList = sorted(list(DQMap.keys()))
    
    DQCount = len(DQList)
    BurstCount = len(BurstList)
    if DQCount > 1:
        maxDqDistance = DQList[DQCount - 1] - DQList[0]
        minDQDistance = 8
        for i in range(1, DQCount):
            
            minDQDistance = min(minDQDistance, DQList[i] - DQList[i - 1])
    else:
        maxDqDistance = -1
        minDQDistance = -1
        
    if BurstCount > 1:
        maxBurstDistance = BurstList[BurstCount - 1] - BurstList[0]
        minBurstDistance = 8
        for i in range(1, BurstCount):
            minBurstDistance = min(minBurstDistance, BurstList[i] - BurstList[i - 1])
    else:
        maxBurstDistance = -1
        minBurstDistance = -1

    
    return [adjDqCount, maxDqDistance, minDQDistance,DQCount, maxBurstDistance, minBurstDistance, BurstCount]
        



      
def processDimm(id, q, dimmList, leadTime):
    for dimm in dimmList:
        # print(dimm)
        errorFile = os.path.join(SPLIT_DATA_PATH, dimm, dimm+"_error.csv")
        # 生成静态信息
        baseSample = getBaseSample(dimm, errorFile)
        
        
       
        df = pd.read_csv(errorFile, low_memory=False)
        df['err_time'] = pd.to_datetime(df['err_time'], format="%Y-%m-%d %H:%M:%S")
        
        UEFlag = False
        firstUER = datetime.now().replace(year=2099)
        UEDf = df[(df['err_type'].isin(UETypeList))].reset_index(drop=True)
        if UEDf.shape[0] != 0:
            UEFlag = True
            firstUER = UEDf.loc[0, 'err_time']
        CEDf = df[(df['err_time'] <  firstUER)].reset_index(drop=True)

        
        CECount =CEDf.shape[0]
        if CECount == 0:
            continue

        # 故障记录
        sampleList = []
        indexSet = set() 
        for  index, error in CEDf.iterrows():
            rowId, columnId, bankId, bankgroupId,rankId =  error['row'], error['column'], error['bank'], error['bankgroup'],error['rank']
            parity = error['RetryRdErrLogParity']

            if rowId == -1:
                continue 

            index =  "{}{}{}{}{}{}".format(rowId, columnId, bankId, bankgroupId,rankId, parity)
            index =  "{}{}{}{}{}".format(rowId, columnId, bankId, bankgroupId,rankId)
            if index  not in indexSet:
                indexSet.add(index)
                continue
            
            # indexSet.add(index)
            
            adjDqCount, maxDqDistance, minDQDistance,DQCount, maxBurstDistance, minBurstDistance, BurstCount = parseBitError(parity)
            
            sample = copy.copy(baseSample)
            sample['adjDqCount'] = adjDqCount
            sample['maxDqDistance'] = maxDqDistance
            sample['minDQDistance'] = minDQDistance
            sample['DQCount'] = DQCount
            sample['maxBurstDistance'] = maxBurstDistance
            sample['minBurstDistance'] = minBurstDistance
            sample['BurstCount'] = BurstCount

            
            sample['label'] = UEFlag


            sampleList.append(sample)
            
        q.put([True, sampleList])   
            
            

def mergeFunction(q):
    writer = get_writer(os.path.join(DATA_SET_PATH,dataSetFile))
    while True:
        [flag , sampleList] = q.get()
        if not flag:
            break
        [writer.writerow(sample) for sample in sampleList]
        
def genDataSet(leadTime):
    
    # DATA_SET_PATH = 'train'
    if not os.path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)
    
    dimmList = os.listdir(SPLIT_DATA_PATH)
    # dimmList = dimmList[10:103]
    q = Queue()
    processList = []
    cpuCount = os.cpu_count() * 2
    # cpuCount = 1
    subListSize = math.ceil(len(dimmList) / cpuCount)
    for i in range(cpuCount):
        subDimm = dimmList[i*subListSize:(i + 1)*subListSize]
        processList.append(Process(target=processDimm, args=(i,q, subDimm, leadTime)))
        
    pMerge = Process(target=mergeFunction, args=[q])
    pMerge.start()
    for p in processList:
        p.start()

    for p in processList:
        p.join()
    q.put([False,[]])
    pMerge.join()
    
    trainFile = os.path.join(DATA_SET_PATH, dataSetFile)
    trainDf = pd.read_csv(trainFile, low_memory=False)


    for item in STATIC_ITEM:
        trainDf["{}".format(item)] = pd.Categorical(pd.factorize(trainDf[item])[0])

    trainDf.to_csv(trainFile, index= False)


print("生成提前预测时间为{}的数据集".format(LEAD))
genDataSet(LEAD)

    

    