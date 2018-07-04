from numpy import *

from kNN_classify import classify0


def file2matrix(filename):
    """
       Desc:
           导入训练数据
       parameters:
           filename: 数据文件路径
       return:
           数据矩阵 returnMat 和对应的类别 classLabelVector
       """
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)


def datingClassTest():
    '''
    Classifier testing code for dating site
    :return:
    '''
    hoRatio = 0.10


    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: ",classifierResult," the real answer is: " , datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: " , (errorCount / float(numTestVecs)))


#datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person: ",resultList[classifierResult - 1])


classifyPerson()