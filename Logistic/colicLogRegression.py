# 利用随机梯度下降预测病马死亡率
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# 函数说明：sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 函数说明：批梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 函数说明：随机梯度上升算法
def stocGradAscent(dataMat,labelMat,numIter=150):
    m,n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+i+j) + 0.1
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

# 函数说明:使用Python写的Logistic分类器做预测
def colicTest():
    trainingSet = []
    trainingLabel = []
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        lineArr = []
        for i in range(len(line)-1):
            lineArr.append(float(line[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(line[-1]))
    # weights = stocGradAscent(np.array(trainingSet),trainingLabel,500)
    weights = gradAscent(np.array(trainingSet), trainingLabel)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        line = line.strip().split('\t')
        numTestVec += 1.0
        lineArr = []
        for i in range(len(line)-1):
            lineArr.append(float(line[i]))
        if int(classifyVector(np.array(lineArr),weights)) != int(line[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)


# 函数说明：分类函数
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 函数说明：使用Sklearn构建Logistic回归分类器
def colicSklearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    # solver:表示梯度优化方法
    classifier = LogisticRegression(solver='liblinear',max_iter=50).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
    colicSklearn()