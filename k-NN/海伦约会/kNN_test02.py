import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import operator

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

# 函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
def file2matrix(filename):
    f = open(filename)
    # readlines()方法读取整个文件所有行，保存在一个列表(list)变量中
    arraylines = f.readlines()
    num_lines = len(arraylines)
    data_Mat = np.zeros((num_lines,3))
    labels = []
    # 行索引
    index = 0

    for lines in arraylines:
        line = lines.strip()
        list_line = line.split('\t')
        data_Mat[index,:] = list_line[0:3]

        if list_line[-1] == 'didntLike':
            labels.append(1)
        elif list_line[-1] == 'smallDoses':
            labels.append(2)
        elif list_line[-1] == 'largeDoses':
            labels.append(3)
        index += 1

    return data_Mat,labels


# 函数说明:可视化数据
def showdatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

# 函数说明:对数据进行归一化处理
def autonorm(data_Mat):
    # 获取数据的最小值,axis=0:每列的最小值
    minVals = data_Mat.min(0)
    maxVals = data_Mat.max(0)

    ranges = maxVals - minVals
    m = data_Mat.shape[0]
    normDataSet = data_Mat - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))

    return normDataSet,ranges,minVals

# 函数说明:测试分类器
def datingClassTest():
    filename = 'datingTestSet.txt'
    data_Mat,data_label = file2matrix(filename)
    normDataSet,ranges,minVals = autonorm(data_Mat)
    m = normDataSet.shape[0]
    ratio = 0.1
    num_test = int(0.1 * m)
    error_count = 0.0
    for i in range(num_test):
        class_result = classification(normDataSet[i,:],normDataSet[num_test:m,:],data_label[num_test:m],4)
        print("分类结果:%d\t真实类别:%d" % (class_result,data_label[i]))
        if class_result != data_label[i]:
            error_count += 1.0
    print("错误率:%f%%" % (error_count/float(num_test) * 100))

# 函数说明:通过输入一个人的三维特征,进行分类输出
def classifyPersion():
    result_list = ["讨厌","有点喜欢","非常喜欢"]
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))

    filename = "./datingTestSet.txt"
    datasetMat,dataLabels = file2matrix(filename)
    normMat,ranges,minVals = autonorm(datasetMat)
    input_arr = np.array([ffMiles,precentTats,iceCream])
    norminput_arr =  (input_arr-minVals)/ranges
    class_result = classification(norminput_arr,normMat,dataLabels,3)
    print("你可能%s这个人" % (result_list[class_result - 1]))

if __name__ == '__main__':
    # datingClassTest()
    # filename = 'datingTestSet.txt'
    # data_Mat,data_label = file2matrix(filename)
    # normDataSet,ranges,minVals = autonorm(data_Mat)
    # showdatas(data_Mat,data_label)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)
    classifyPersion()









