import pandas as pd
import seaborn as sns
from config import *
import matplotlib.pyplot as plt 
# 构造数据


SP_PATH = os.path.join(PIC_PATH,'sp')
if not os.path.exists(SP_PATH):
    os.makedirs(SP_PATH)
    

# path = '/home/hw-admin/yixuan/featureAnalysis/data_set'
path = '/home/hw-admin/yixuan/featureAnalysis/time'
subDatesetList = os.listdir(path)
count = len(subDatesetList)

itemMap = {}
itemList = patternList
for item in itemList:
    itemMap[item] = []

for i in range(count):
    fileName = os.path.join(path, "{}.csv".format(i))
    # fileName = os.path.join(path,subDatesetList[i])
    df = pd.read_csv(fileName)
    for item in itemList:
        subDf = df[[item,'label']]
        corr = subDf.corr(method='spearman')
        itemMap[item].append(corr.iloc[0,1])
        # print(item, corr.iloc[0,1])
    
indexList = []
for i in range(count):
    indexList.append(i)
for item in itemList:
    x = indexList
    y = itemMap[item]
    plt.ylim(-1.0, 1.0)
    plt.plot(x, y)
    plt.xlabel("month")
    plt.ylabel("correlation coefficient")
    plt.axhline(0, color='r', linestyle='--', label='Mean')
    plt.title(item)
    plt.savefig(os.path.join(SP_PATH, "{}.png".format(item)))
    plt.cla()
# # 绘制相关系数热力图
sns.heatmap(corr, annot=True, cmap="YlGnBu")

