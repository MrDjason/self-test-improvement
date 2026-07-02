from math import log
import operator

# 创建dataSet
def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # featVec:[0, 1, 'no']
        currentLabel = featVec[-1] # currentLabel:no
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # labelCounts:{'yes': 2, 'no': 3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # probability概率 = 出现次数 / 总数
        shannonEnt -= prob * log(prob, 2) # 香农熵计算公式
    print(shannonEnt)
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    '''
    dataset:待划分数据集
    axis:划分数据集的特征
    value:需要返回特征值
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # 赋值从开头到axis
            reducedFeatVec.extend(featVec[axis+1:]) # axis之后列表数据进行合并
            retDataSet.append(reducedFeatVec) # 将axis=1的行提取并且取出第axis的列返回
    print(retDataSet)
    return retDataSet
    
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 去除标签列，计算有多少特征
    baseEntropy = calcShannonEnt(dataSet) # 计算原始数据的香农熵
    bestInfoGain, bestFeature = 0.0, -1 # 初始化最好结果
    for i in range(numFeatures): # 遍历每一个特征 
        featList = [example[i] for example in dataSet] # 取出当前特征的所有值 并约束特征范围
        uniqueVals = set(featList) # 去重
        newEntropy = 0.0
        for value in uniqueVals: # 分别选取特征0/1进行划分
            subDataSet = splitDataSet(dataSet, i, value) # 切分数据
            prob = len(subDataSet) / float(len(dataSet)) # 计算这个子集占多大概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算加权熵
        infoGain = baseEntropy - newEntropy # 计算信息增益
        if (infoGain > bestInfoGain): # 保留最大信息增益和对应特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature # 返回最优编号

# 递归构建决策树
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), # classCount.iteritems()将字典变成键值对列表，每一对都是[(类别,次数)]的列表中的元组
                              key = operator.itemgetter(1), # 用operator中的itemgetter获取第1个元素，按第二列次数排列
                              reverse = True) # 倒序排列，大的在前
    # [('no', 3), ('yes', 2)]
    return sortedClassCount[0][0] # 返回数量最多的那个类别

# 创建树的函数代码
def createTree(dataSet, labels):
    # 递归停止条件
    classList = [example[-1] for example in dataSet] # 拿出所有标签
    if classList.count(classList[0]) == len(classList): # 如果classList所有标签都是相同的
        return classList[0] # 直接返回
    if len(dataSet[0]) == 1: # 每次递归，都会进行特征删除，当只剩标签的时候调用递归构建决策树
        return majorityCnt(classList)
    
    # 选取最好特征
    bestFeat = chooseBestFeatureToSplit(dataSet) # 选择最好的特征编号
    bestFeatLabel = labels[bestFeat] # 拿到最好特征标签的名字
    myTree = {bestFeatLabel:{}} # 创建树的字典
    del(labels[bestFeat]) # 删除已经用过的特征 （不太懂为什么删）
    featValues = [example[bestFeat] for example in dataSet] # 取出特征里所有数据
    uniqueVals = set(featValues) # 去重

    # 递归启动
    for value in uniqueVals: # 对特征的每一个特征值（0、1） 1.切分数据集、2.递归调用createTree、3.把结果存在数字典中
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),\
                                                  subLabels)
    return myTree # 返回建好的树

if __name__ == '__main__':
    mydata, label=createDataSet()
    calcShannonEnt(mydata)
    splitDataSet(mydata, 0, 1)