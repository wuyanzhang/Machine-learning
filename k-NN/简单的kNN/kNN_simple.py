"""
函数说明:创建数据集
Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
"""
import numpy as np
import operator

def createDataset():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classification(inX,dataSet,labels,k):
    # numpy函数shape[0]返回dataSet的行数
    dataSet_col = dataSet.shape[0]

    diffMat = np.tile(inX,(dataSet_col,1)) - dataSet
    sq_diffMat = diffMat**2
    sq_distance = sq_diffMat.sum(axis=1)
    distances = sq_distance**0.5
    sorteddistances_index = distances.argsort()

    # 定义一个记录类别次数的字典
    classCount = {}

    for i in range(k):
        label = labels[sorteddistances_index[i]]
        classCount[label] = classCount.get(label,0) + 1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]




if __name__ == '__main__':
    # 创建数据集
    group,labels = createDataset()

    # 测试集(动作数，接吻数)
    test = [101,20]

    # kNN分类
    test_class = classification(test,group,labels,3)

    # 打印结果
    print(test_class)



























