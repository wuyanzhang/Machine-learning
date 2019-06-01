import numpy as np
import matplotlib.pyplot as plt

# 函数说明：加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line = line.strip().split()
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(int(line[2]))
    fr.close()
    return dataMat,labelMat

# 函数说明：绘制数据集
def plotDataSet():
    dataMat,labelMat = loadDataSet()
    dataMat = np.array(dataMat)
    num = len(dataMat)
    x0 = [];x1 = []
    y0 = [];y1 = []
    for i in range(num):
        if int(labelMat[i]) == 1:
            x0.append(dataMat[i][0])
            y0.append(dataMat[i][1])
        else:
            x1.append(dataMat[i][0])
            y1.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111) # 1*1网格中的第一个子图（只有一个子图）
    ax.scatter(x0,y0,s=20,c='red',marker='s',alpha=.5)
    ax.scatter(x1,y1,s=20,c='green',alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    plotDataSet()