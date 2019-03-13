from sklearn.neighbors import KNeighborsClassifier
import numpy as np


"""
使用传统方式读取数据集
def setdata(filename):
    featmat = []
    labels = []
    with open(filename) as file:
        for line in file.readlines():
            line = line.strip('\n')
            line = line.strip(' ')
            line = line.split('\t')
            featmat.append(list(map(float, line[:-1])))
            labels.append(int(line[-1]))
    return featmat, labels
"""


def npsetdata(filename):
    """numpy加载数据集"""
    file = open('./dataset/date')
    #file = open('dataset/donate_blood')
    arraylines = file.readlines()
    arraylen = len(arraylines)
    returnmat = np.zeros((arraylen, 3))
    labels = []
    index = 0
    for line in arraylines:
        line = line.strip()
        line = line.strip('\t')
        listofline = line.split('\t')
        #listofline = line.split(',')
        returnmat[index, : ] = listofline[:3]
        labels.append(int(listofline[-1]))
        index += 1
    return returnmat, labels


def normmat(mat):
    """归一化数值"""
    minlist = mat.min(0)
    maxlist = mat.max(0)
    normdataset = np.zeros(np.shape(mat))
    m = mat.shape[0]
    normdataset = mat - np.tile(minlist, (m, 1))
    normdataset = normdataset / np.tile((maxlist - minlist), (m, 1))
    return normdataset


def testfunc():
    """测试分类准确度"""
    dataset, label = npsetdata('./dataset/date')
    feat = normmat(dataset)
    #print(feat)
    #print(label)
    neighbor = KNeighborsClassifier(n_neighbors=3)
    neighbor.fit(feat[:680], label[:680])

    true = 0
    flase = 0
    index = 0
    for item in feat[680:]:
        lab = label[680:][index]
        prediction = neighbor.predict([item])
        print("预测结果为：" + str(prediction) + "\t实际结果为：[" + str(lab) + ']')
        if prediction == lab:
            true += 1
        else:
            flase += 1
        index += 1
    result = true / (true + flase)
    print("准确率为：" + str(result))


if __name__ == "__main__":
    testfunc()