import numpy as np

# 函数说明：创建实验样本
def loadDataSet():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    label = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表非侮辱性词汇
    return dataset,label

# 函数说明：建立词汇表
def createVocabList(dataset):
    vocabSet = set([])
    for row in dataset:
        vocabSet = vocabSet | set(row)
    return list(vocabSet)

# 函数说明：根据词汇表，将数据集向量化
def Words2Vec(VocabList,input_data):
    resVec = [0]*len(VocabList)
    for word in input_data:
        if word in VocabList:
            resVec[VocabList.index(word)] = 1
    return resVec

# 函数说明:朴素贝叶斯分类器训练函数
def trainingNB(trainingMat,trainingLabel):
    numTrainingVec = len(trainingMat)
    numWords = len(trainingMat[0])
    pAbusive = sum(trainingLabel) / float(numTrainingVec)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainingVec):
        if trainingLabel[i] == 1:
            p1Num += trainingMat[i]
            p1Denom += sum(trainingMat[i])
        else:
            p0Num += trainingMat[i]
            p0Denom += sum(trainingMat[i])
    p0 = np.log(p0Num / p0Denom)
    p1 = np.log(p1Num / p1Denom)
    return p0,p1,pAbusive

# 函数说明:朴素贝叶斯分类器分类函数
def classifyNB(vec,p0,p1,pAbusive):
    p0 = sum(vec * p0) + np.log(1.0 - pAbusive)
    p1 = sum(vec * p1) + np.log(pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

# 函数说明：测试朴素贝叶斯分类器
def tesingNB():
    dataset,label = loadDataSet()
    myVocabList = createVocabList(dataset)
    trainMat = []
    for row in dataset:
        trainMat.append(Words2Vec(myVocabList,row))
    p0,p1,pAbusive = trainingNB(np.array(trainMat),np.array(label))
    testData = ['stupid', 'garbage']
    testVec = np.array(Words2Vec(myVocabList,testData))
    if classifyNB(testVec,p0,p1,pAbusive):
        print(testData,'属于侮辱类')
    else:
        print(testData,'属于非侮辱类')

if __name__ == "__main__":
    tesingNB()