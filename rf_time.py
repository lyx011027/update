
import os
import random
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay,average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from config import *
from multiprocessing import Process
# from imblearn.combine import SMOTEENN
import lightgbm as lgb
import copy

import xgboost as xgb # XGBoost 包
from xgboost.sklearn import XGBClassifier # 设置模型参数
from sklearn import preprocessing
LEAD = timedelta(minutes=0)
threshold = 0.2
Trian = 0.7

def getDynamicTrainSample():
    sample = {}
    sample = getFrequencySample(sample)
    # sample = getBitLevelSample(sample)
    sample = getSubBankSample(sample)
    sample = getCECountSample(sample)

    return sample

dynamicItem = list(getDynamicTrainSample().keys())
trainItem = ([]
+ dynamicItem
# + STATIC_ITEM
# +['time']
)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(PIC_PATH):
    os.makedirs(PIC_PATH)


def plot_feature_importances(feature_importances,title,feature_names, picFile):
#    将重要性值标准化
    feature_importances = 100.0*(feature_importances/max(feature_importances))
    # index_sorted = np.flipud(np.argsort(feature_importances)) #上短下长
    #index_sorted装的是从小到大，排列的下标
    index_sorted = np.argsort(feature_importances)# 上长下短
#    让X坐标轴上的标签居中显示
    bar_width = 1
    # 相当于y坐标
    pos = np.arange(len(feature_importances))+bar_width/2
    plt.figure(figsize=(16,4))
    # plt.barh(y,x)
    plt.barh(pos,feature_importances[index_sorted],align='center')
    # 在柱状图上面显示具体数值,ha参数控制参数水平对齐方式,va控制垂直对齐方式
    for y, x in enumerate(feature_importances[index_sorted]):
        plt.text(x+2, y, '%.4s' %x, ha='center', va='bottom')
    plt.yticks(pos,feature_names[index_sorted])
    
    for i in range(len(index_sorted) - 10, len(index_sorted)):
        idex = index_sorted[i]
        print(feature_names[idex])
    plt.title(title)
    plt.savefig(picFile,dpi=1000)
    plt.clf()
    plt.cla()
    



def trainAndTest(index, trainDf, testDf, trainItem):
    global FPMap 
    global TPMap 
    global PositiveCount 
    global lastPrecision
    global lastPeriodDIMMUE
    global MostPeriodDIMMUE
    global lastPeriodDIMMCE
    global MostPeriodDIMMCE
    global threshold
    X_train , Y_train = trainDf[trainItem].fillna(-1), trainDf['label'].fillna(False)
    X_test , Y_test = testDf[trainItem].fillna(-1), testDf['label'].fillna(False)
    
    trainSnList = trainDf['dimm_sn'].tolist()
    testSnList = testDf['dimm_sn'].tolist()
    testTrueSnList = testDf[testDf['label'] == True]['dimm_sn'].drop_duplicates().tolist()
    testFalseSnList = testDf[testDf['label'] == False]['dimm_sn'].drop_duplicates().tolist()
    

    # 训练模型
    rfc = RandomForestClassifier()
    
    # rfc = lgb.LGBMClassifier(force_col_wise=True)
    
    rfc = XGBClassifier(
    learning_rate =0.1,
    n_estimators=50,
    max_depth=10,
    min_child_weight=3,
    gamma=3.5,
    subsample=0.5,
    colsample_bytree=0.5,
    objective= 'binary:logistic',#'multi:softprob'
    # num_class=3, #    'num_class':3, #类别个数
    nthread=24,
    scale_pos_weight=1,
    seed=42)
    
    rfc.fit(X_train, Y_train)
    # 输出并保存 feature importance
    picFile = os.path.join(PIC_PATH, "{}-importance.png".format(index))
    # for i in range (len(trainItem)):
    #     print(trainItem[i], rfc.feature_importances_[i])
    trainItem = np.array(trainItem)
    
    plot_feature_importances(rfc.feature_importances_, "feature importances", trainItem,picFile)
    # 输出对应 threshold的结果

    predicted_proba = rfc.predict_proba(X_test)
    if MostPeriodDIMMUE != 0:
        UEp = (MostPeriodDIMMUE - lastPeriodDIMMUE)/MostPeriodDIMMUE
        CEp = (MostPeriodDIMMCE - lastPeriodDIMMCE)/MostPeriodDIMMCE
        threshold = max(0.2,0.5*UEp*(1-CEp))
        threshold = max(0.2,0.5*UEp)
    print(threshold)
    Y_pred = (predicted_proba [:,1] >= threshold).astype('int')
    # Y_pred = rfc.predict(X_test) 
    print("\nModel used is: Random Forest classifier") 
    acc = accuracy_score(Y_test, Y_pred) 
    print("The accuracy is {}".format(acc))
    prec = precision_score(Y_test, Y_pred) 
    print("The precision is {}".format(prec)) 
    rec = recall_score(Y_test, Y_pred) 
    print("The recall is {}".format(rec)) 
    f1 = f1_score(Y_test, Y_pred) 
    print("The F1-Score is {}".format(f1)) 
    
    length = len(testSnList)

    for i in range(length):
        if Y_pred[i] == 1 and Y_test[i] == False:
            FPMap[testSnList[i]] = True
            
        if Y_pred[i] == 1 and Y_test[i] == True:
            TPMap[testSnList[i]] = True
    for i in range(length):
        if Y_test[i] == True and testSnList[i] in FPMap:
            del FPMap[testSnList[i]]
            TPMap[testSnList[i]] = True
    
    FPCount = len(FPMap)
    TPCount = len(TPMap)
    PositiveCount += len(testTrueSnList)
    lastPeriodDIMMUE=  len(testTrueSnList)
    MostPeriodDIMMUE= max(MostPeriodDIMMUE, len(testTrueSnList))
    lastPeriodDIMMCE=  len(testFalseSnList)
    MostPeriodDIMMCE= max(MostPeriodDIMMCE, len(testFalseSnList))
    
    print("predict time: {} , FPCount: {} , TPCount: {} , PositiveCount:{}".format(
        index, FPCount, TPCount, PositiveCount))    
    if PositiveCount != 0 and TPCount != 0:
        precision = TPCount/(FPCount + TPCount)
        recall = TPCount/(PositiveCount)
        F1_score = (2 * precision * recall)/(precision + recall)
        print('precision: {} , recall: {} , F1-score: {}'.format(precision,recall, F1_score))
    lastPrecision = TPCount/(FPCount + TPCount)
    prec, recall, _ = precision_recall_curve(Y_test, predicted_proba [:,1], pos_label=1)
    pr_display = PrecisionRecallDisplay(estimator_name = 'rf',precision=prec, recall=recall, average_precision=average_precision_score(Y_test, predicted_proba [:,1], pos_label=1))
    pr_display.average_precision
    pr_display.plot()
    plt.xlim(0.0, 1.1)
    plt.ylim(0.0, 1.1)
    plt.savefig(os.path.join(PIC_PATH,'{}-p-r.png'.format(index)),dpi=1000)
    plt.cla()
    
    # 保存模型
    with open(os.path.join(MODEL_PATH,'{}.pkl'.format(index)), 'wb') as fw:
        pickle.dump(rfc, fw)

def getTrainDf(startIndex, endIndex):
    path = DATA_SET_PATH


    df = pd.read_csv(os.path.join(path, "0.csv"))
    for i in range(max(1,startIndex), endIndex):
        fileName = os.path.join(path, "{}.csv".format(i))
        subDf = pd.read_csv(fileName)
        df = pd.concat([df, subDf]).reset_index(drop=True)
        
    return df
FPMap = {}
TPMap = {}
PositiveCount = 0
lastPrecision = 0.5
lastPeriodDIMMUE = 0
MostPeriodDIMMUE = 0
lastPeriodDIMMCE = 0
MostPeriodDIMMCE = 0
DATA_SET_PATH = 'time'
path = DATA_SET_PATH
subDatesetList = os.listdir(path)
count = len(subDatesetList)
# 生成初始训练集
# 初始训练集大小
initialDatasetSize = 12

for i in range(initialDatasetSize, count):
    trainDf = getTrainDf(i-initialDatasetSize,i)
    testDf = pd.read_csv(os.path.join(path, "{}.csv".format(i)))
    trainAndTest(i,trainDf, testDf, trainItem)
    # df = pd.concat([df, testDf])

# trainDf = getTrainDf(0,initialDatasetSize)
# testDf = getTrainDf(initialDatasetSize,count)
# trainAndTest(0,trainDf, testDf, trainItem)
