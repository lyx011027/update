from datetime import timedelta
import os
# 存放原始数据的文件夹
DATA_SOURCE_PATH = "/home/hw-admin/yixuan/data"
# 按sn号切分
SPLIT_DATA_PATH = os.path.join("split")
# 存放生成数据集的路径
DATA_SET_PATH = os.path.join("data_set")
# 存放测试模型的路径
TEST_MODEL_PATH = os.path.join("model_test")
# 存放测试模型R-P曲线图的文件夹
TEST_PIC_PATH = os.path.join("pic_test")
# 存放训练得到的数据集
MODEL_PATH =  os.path.join("model")
# 存放R-P曲线图的文件夹
PIC_PATH = os.path.join("pic")
# 提前预测时间，单位为minute
AHEAD_TIME_List = [timedelta(seconds=15),timedelta(minutes=1),timedelta(minutes=15),timedelta(minutes=30),timedelta(minutes=60), timedelta(hours=6)]
# 按batch生成数据集时，batch中dimm的数量，如果使用sample_batch.py生成数据集时发生OOM，则降低该值
BATCH_SIZE = 10000
MAXIMUM_RATIO = 100
STATIC_ITEM = [ "bit_width_x" ,"capacity"  ,"dimm_part_number"   ,"procedure" ,"rank_count" ,
               "speed" ,
               "vendor"]


sampleDistance = 5
CETypeList = ['CE']
UERTypeList = ['UCE']
UEOTypeList = []
UETypeList = UERTypeList + UEOTypeList

PatrolScrubbingUETypeList = ['Downgraded Uncorrected PatrolScrubbing Error']

OBSERVATION_TIME_LIST = [timedelta(minutes=6), timedelta(hours=6), timedelta(hours=24), timedelta(hours=72), timedelta(hours=120)]
# OBSERVATION_TIME_LIST = [timedelta(minutes=1), timedelta(minutes=5), timedelta(hours=1), timedelta(hours=3), timedelta(hours=12), timedelta(hours=24)]

# 提前预测时间
LEAD_TIME_LIST = [timedelta(seconds=1),timedelta(seconds=30),timedelta(minutes=1),timedelta(minutes=5),timedelta(minutes=60)]


def getMinutes(time):
    return int(time.days * 24 * 60 + time.seconds / 60)
CEIntervalNumList = [3, 5, 7] 
FltCnt = {'Cell':2,'Row':2,'Column':2,'Bank':3,'Device':2}


def getDynamicSample():
    sample = {}
    sample = getBitLevelSample(sample)

    return sample



patternList = ['adjDqCount', 'maxDqDistance', 'minDQDistance','DQCount', 'maxBurstDistance', 'minBurstDistance', 'BurstCount']
def getBitLevelSample(sample):
    for pattern in patternList:
        sample[pattern] = -1
    return sample


      
dynamicItem = list(getDynamicSample().keys())
LEAD = timedelta(minutes=0)
dataSetFile = "{}_30days.csv".format(sampleDistance)

subBankTime = timedelta(minutes=5)
OBSERVATION = timedelta(hours=120)
Predict = timedelta(days=30)
Interval = timedelta(minutes=5)

