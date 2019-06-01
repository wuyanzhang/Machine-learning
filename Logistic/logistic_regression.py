import numpy as np
import matplotlib.pyplot as plt

# 函数说明：加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

def loadDataSet1():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line = line.strip().split()
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(int(line[2]))
    fr.close()
    return dataMat,labelMat

# 函数说明：sigmoid函数
def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))

# 函数说明：梯度上升算法
def gradAscent(dataMat,labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m,n = np.shape(dataMat)
    alpha = 0.001
    maxstep = 500
    weights = np.ones([n,1])
    for k in range(maxstep):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
    return weights.getA()

# 函数说明：绘制最优分界线
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet1()
    dataMat = np.array(dataMat)
    num = len(dataMat)
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(num):
        if int(labelMat[i]) == 1:
            x0.append(dataMat[i][0])
            y0.append(dataMat[i][1])
        else:
            x1.append(dataMat[i][0])
            y1.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 1*1网格中的第一个子图（只有一个子图）
    ax.scatter(x0, y0, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(x1, y1, s=20, c='green', alpha=.5)

    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    # print(gradAscent(dataMat,labelMat))
    plotBestFit(weights)
