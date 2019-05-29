from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pickle

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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
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

# 函数说明:统计LabelList中出现此处最多的元素(类标签)
def majorityCnt(LabelList):
    classCount = {}
    for label in LabelList:
        if label not in classCount.key():
            classCount[label] = 0
        classCount[label] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


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
        # print("第%d个特征的信息增益为：%.3f" % (i,InfoGain))
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeatureIndex = i
    return bestFeatureIndex

def CreateTree(dataset,labels,featLabels):
    LabelList = [example[-1] for example in dataset]
    if LabelList.count(LabelList[0]) == len(dataset):
        return LabelList[0]
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majorityCnt(LabelList)
    bestFeatureIndex = chooseBestFeature(dataset)
    bestFeatureLabel = labels[bestFeatureIndex]
    featLabels.append(bestFeatureLabel)
    mytree = {bestFeatureLabel:{}}
    del(labels[bestFeatureIndex])
    FeatureList = [example[bestFeatureIndex] for example in dataset]
    UniqueFeatureList = set(FeatureList)
    for value in UniqueFeatureList:
        mytree[bestFeatureLabel][value] = CreateTree(splitDataSet(dataset,bestFeatureIndex,value),labels,featLabels)
    return mytree

# 函数说明:使用决策树分类
def classify(inputTree,featLabels,testVec):
    firstNode = next(iter(inputTree))
    secondDict = inputTree[firstNode]
    featIndex = featLabels.index(firstNode)
    for key in secondDict:
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                result = classify(secondDict[key],featLabels,testVec)
            else:
                result = secondDict[key]
    return result

# 函数说明:存储决策树
def storeTree(inputTree,filename):
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)

# 函数说明:读取决策树
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

# 函数说明:获取决策树叶子结点的数目
def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                               #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                               #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':              #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

# 函数说明:获取决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            #更新层数
    return maxDepth

# 函数说明:绘制结点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

# 函数说明:标注有向边属性值
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

# 函数说明:绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


# 函数说明:创建绘制面板
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                              #创建fig
    fig.clf()                                                                     #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                   #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                               #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                              #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                    #x偏移
    plotTree(inTree, (0.5,1.0), '')                                              #绘制决策树
    plt.show()

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = CreateTree(dataSet,labels,featLabels)
    storeTree(myTree,'StorageTree.txt')
    # grabTree('StorageTree.txt')
    # print(myTree)
    # createPlot(myTree)
    testVec = [0,1]
    result = classify(myTree,featLabels,testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')

