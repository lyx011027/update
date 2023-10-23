
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
threshold = 0.4
Trian = 0.7

def getDynamicTrainSample():
    sample = {}
    sample = getBitLevelSample(sample)

    
    return sample

dynamicItem = list(getDynamicTrainSample().keys())
trainItem = dynamicItem

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



trainFile = os.path.join(DATA_SET_PATH, dataSetFile)
    
trainDf = pd.read_csv(trainFile, low_memory=False)


testDf =  copy.copy(trainDf)

def trainAndTest(time,trainItem):

    
    print("提前预测时间 = {}".format(time))
    
    # 提取正样本
    true_sn = trainDf[trainDf['label'] == True]['dimm_sn'].drop_duplicates().tolist()    
    true_sn_train = random.sample(true_sn, int(len(true_sn)*Trian))
    # 提取训练集正样本
    true_df_train = trainDf[trainDf['dimm_sn'].isin(true_sn_train)]
    # 提取测试集正样本
    true_df_test = testDf[(~testDf['dimm_sn'].isin(true_sn_train)) & (testDf['dimm_sn'].isin(true_sn))]

    false_sn = trainDf[~trainDf['dimm_sn'].isin(true_sn)]['dimm_sn'].drop_duplicates().tolist()
    
    # 提取负样本
    false_sn_train = random.sample(false_sn, int(len(false_sn) *Trian))
    # false_sn_train = random.sample(false_sn, len(true_sn_train) * 10)
    
    # 生成训练集负样本data与label
    false_df_train = trainDf[trainDf['dimm_sn'].isin(false_sn_train)]
    # 生成测试集负样本data与label
    false_df_test = testDf[(~testDf['dimm_sn'].isin(false_sn_train)) & (testDf['dimm_sn'].isin(false_sn))]

    print(len(true_sn_train),len(true_sn) - len(true_sn_train) , len(false_sn_train),len(false_sn) - len(false_sn_train))
    # 生成训练集
    train = pd.concat([true_df_train, false_df_train]).reset_index(drop=True)
    
    X_train , Y_train = train[trainItem].fillna(-1), train['label'].fillna(False)
    

    
    # 生成测试集
    test =  pd.concat([true_df_test, false_df_test])
    testSnList = test['dimm_sn'].tolist()
    X_test = test.fillna(-1)[trainItem]
    # 连接训练集与测试集的label
    Y_test = test['label'].reset_index(drop=True).fillna(False)

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
    picFile = os.path.join(PIC_PATH, "{}-importance.png".format(time))
    # for i in range (len(trainItem)):
    #     print(trainItem[i], rfc.feature_importances_[i])
    trainItem = np.array(trainItem)
    
    plot_feature_importances(rfc.feature_importances_, "feature importances", trainItem,picFile)
    # 输出对应 threshold的结果

    predicted_proba = rfc.predict_proba(X_test)
    

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
    # p.append(prec)
    # r.append(rec)
    # f.append(f1)
    length = len(testSnList)
    FPMap = {}
    for i in range(length):
        if Y_pred[i] == 1 and Y_test[i] == 0:
            if testSnList[i] not in FPMap:
                FPMap[testSnList[i]] = 0
            FPMap[testSnList[i]] += 1
    TPMap = {}
    for i in range(length):
        if Y_pred[i] == 1 and Y_test[i] == 1:
            if testSnList[i] not in TPMap:
                TPMap[testSnList[i]] = 0
            TPMap[testSnList[i]] += 1
            
    
    FP ,TP,= 0,0
    for i in FPMap.values():
        if i >= 10 :
            FP += 1
    for i in TPMap.values():
        if i >= 10:
            TP += 1
    
    print("ahead time: {} , FP: {} , TP: {} , precision: {} , recall: {}".format(time, FP, TP, TP/(FP + TP), TP/(len(true_sn) - len(true_sn_train))))       

    p.append(TP/(FP + TP))
    r.append(TP/(len(true_sn) - len(true_sn_train)))
    # print(TPMap.keys())
    prec, recall, _ = precision_recall_curve(Y_test, predicted_proba [:,1], pos_label=1)
    pr_display = PrecisionRecallDisplay(estimator_name = 'rf',precision=prec, recall=recall, average_precision=average_precision_score(Y_test, predicted_proba [:,1], pos_label=1))
    pr_display.average_precision
    pr_display.plot()
    plt.xlim(0.0, 1.1)
    plt.ylim(0.0, 1.1)
    plt.savefig(os.path.join(PIC_PATH,'{}-p-r.png'.format(time)),dpi=1000)
    plt.cla()
    
    # 保存模型
    with open(os.path.join(MODEL_PATH,'{}.pkl'.format(time)), 'wb') as fw:
        pickle.dump(rfc, fw)
p = []
r = []
f = []
for i in range(10):
    trainAndTest(LEAD,trainItem)
print(p, sum(p) / len(p))
print(r, sum(r) / len(r))
# print(f, sum(f) / len(f))

