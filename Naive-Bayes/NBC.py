# -*- coding: UTF-8 -*-
import re
import numpy as np
import random

# 函数说明：接收一个大字符串并将其解析为字符串列表
def textParse(bigstring):
    List = re.split(r'\W+',bigstring)
    return [word.lower() for word in List if len(word) > 2]

# 函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
def createVocabList(dataset):
    vocabset = set([])
    for word in dataset:
        vocabset = vocabset | set(word)
    return list(vocabset)

# 函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
def word2Vec(vocabList,inputSet):
    resultMat = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            resultMat[vocabList.index(word)] += 1
    return resultMat

# 函数说明:朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

# 函数说明:朴素贝叶斯分类器分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def spamTest():
    datasetList = []
    labelList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        datasetList.append(wordList)
        labelList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        datasetList.append(wordList)
        labelList.append(0)
    vocabList = createVocabList(datasetList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainLabel = []
    for i in trainingSet:
        trainMat.append(word2Vec(vocabList,datasetList[i]))
        trainLabel.append(labelList[i])
    p0,p1,pSpam = trainNB0(np.array(trainMat),np.array(trainLabel))
    errorCount = 0
    for docIndex in testSet:
        wordVector = word2Vec(vocabList, datasetList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0, p1, pSpam) != labelList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", datasetList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    spamTest()


