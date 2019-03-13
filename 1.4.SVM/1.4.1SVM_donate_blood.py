from sklearn import svm
import numpy as np


def loaddata(filename):
    """numpy读取数据集"""
    f = open('dataset/donate_blood')
    arraylines = f.readlines()
    arraylen = len(arraylines)
    returnMat = np.zeros((arraylen, 4))
    labels = []
    index = 0
    for line in arraylines:
        line = line.strip()
        line = line.strip('\t')
        listofline = line.split(',')
        returnMat[index, : ] = listofline[:4]
        labels.append(int(listofline[-1]))
        index += 1
    return returnMat, labels


def normlist(dataset):
    """归一化数值"""
    minlst = dataset.min(0)
    maxlst = dataset.max(0)
    #print(minlst, maxlst)
    normdataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minlst, (m, 1))
    normdataset = normdataset / np.tile((maxlst - minlst), (m, 1))
    return normdataset


def testfunc():
    """测试分类准确度"""
    dataset, label = loaddata('./dataset/date')
    feat = normlist(dataset)
    #print(feat)
    #print(label)
    SVM = svm.SVC() # kernel='rbf' kernel='linear'可选核函数
    SVM.fit(feat[:620], label[:620])

    true = 0
    flase = 0
    index = 0
    for item in feat[620:]:
        lab = label[620:][index]
        prediction = SVM.predict([item])
        print("预测结果为：" + str(prediction) + "\t实际结果为：[" + str(lab) + ']')
        if prediction == lab:
            true += 1
        else:
            flase += 1
        index += 1
    result = true / (true + flase)
    print("准确率为：" + str(result))


if __name__ == "__main__":
    returnmat, labels = loaddata('dataset/date')
    #print(returnmat)
    #print(labels)

    normdataset = normlist(returnmat)
    #print(normdataset)

    testfunc()