from math import log

# 函数说明：计算给定数据集的经验熵（香农熵）
def calcShannonEnt(dataset):
    row_num = len(dataset)
    LabelCount = {}
    shannonEnt = 0.0
    for row in dataset:
        label = row[-1]
        if label not in LabelCount.keys():
            LabelCount[label] = 0
        LabelCount[label] += 1
    for key in LabelCount:
        prob = float(LabelCount[key]) / row_num
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 函数说明：创建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']
    return dataSet, labels

# 函数说明:按照给定特征划分数据集
def splitDataSet(dataset,axis,value):
    ReducedDataSet = []
    for row in dataset:
        if row[axis] == value:
            cur = row[:axis]
            cur.extend(row[axis+1:])
            ReducedDataSet.append(cur)
    return ReducedDataSet

# 函数说明:选择最优特征（包含了计算信息增益）
def chooseBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeatureIndex = -1

    for i in range(numFeatures):
        FeatureList = [example[i] for example in dataset]
        UniqueFeatureList = set(FeatureList)
        newEntropy = 0.0 # 条件熵
        for value in UniqueFeatureList:
            CurDataSet = splitDataSet(dataset,i,value)
            prob = len(CurDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(CurDataSet)
        InfoGain = baseEntropy - newEntropy
        print("第%d个特征的信息增益为：%.3f" % (i,InfoGain))
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeatureIndex = i
    return bestFeatureIndex

if __name__ == "__main__":
    dataset,features = createDataSet()
    print("最优特征索引值：" + str(chooseBestFeature(dataset)))



















