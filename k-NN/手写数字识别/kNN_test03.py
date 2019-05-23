import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier as KNN

# 函数说明:将32x32的二进制图像转换为1x1024向量
def img2vector(filename):
    vector = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector

# 函数说明:手写数字分类测试
def handwritingClassTest():
    labels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        filename = trainingFileList[i]
        classNumber = int(filename.split('_')[0])
        labels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (filename))
    neigh = KNN(n_neighbors=3,algorithm='auto')
    neigh.fit(trainingMat,labels)

    testingFileList = os.listdir("testDigits")
    n = len(testingFileList)
    errorCount = 0.0
    for i in range(n):
        filename = testingFileList[i]
        classNumber = int(filename.split('_')[0])
        vector_test = img2vector('testDigits/%s' % (filename))
        classifyResult = neigh.predict(vector_test)
        print("分类返回结果为%d\t真实结果为%d" % (classifyResult, classNumber))
        if (classifyResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / n * 100))

if __name__ == "__main__":
    handwritingClassTest()