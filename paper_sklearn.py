import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv
import pandas as pd

def createStockCodeList():
    ScodeList = []
    with open('/Users/gongxing/Desktop/data/shsz20.csv', encoding='gb2312') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            ScodeList.append(line[0])

    return ScodeList

def loadDataSet(fileName, network):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    with open(fileName, encoding="gb2312") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                if len(line) == 19:
                    if network == 0:
                        dataMat.append(
                            [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
                             float(line[6]),
                             float(line[7]),
                             float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12])])
                    else:
                        dataMat.append(
                            [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
                             float(line[6]),
                             float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]),
                             float(line[12]), float(line[13]), float(line[14]), float(line[15]), float(line[16]),
                             float(line[17])])
                    labelMat.append(int(line[18]))
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    return dataMat, labelMat

def loadPriceList(fileName, YLength, PredictLength):
    PriceList = []
    with open(fileName, encoding="gb2312") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            PriceList.append(float(line[2]))

    return PriceList[-(YLength+PredictLength):]

def TrainTestSplit(Xdata,Ydata, TestNum):
    length = len(Xdata)
    TrianNum = length-TestNum
    X_train = Xdata[TrianNum-360:TrianNum]
    X_train = np.array(X_train)
    Y_train = Ydata[TrianNum-360:TrianNum]
    Y_train = np.array(Y_train)
    X_test = Xdata[-TestNum:]
    X_test = np.array(X_test)
    Y_test = Ydata[-TestNum:]
    Y_test = np.array(Y_test)
    return X_train, Y_train, X_test, Y_test


def backAccount(SignalList, PriceList, Sum):
    # 持仓
    Position = [0, 0]
    # 账户
    Account = Sum * 1
    for i in range(len(SignalList)):
        if SignalList[i] == 1:
            if Position[0] == 0:
                Buy_Amount = int(Account / (100 * PriceList[i]))
                Position = [PriceList[i], Buy_Amount]
                Account = Account - 100 * PriceList[i] * Buy_Amount
        else:
            if Position[0] != 0:
                Account = Account + 100 * Position[0] * Position[1]
                Position = [0, 0]
    # print(Position)
    if Position[0] != 0:
        Account = Account + 100 * PriceList[-1] * Position[1]

    print(Account/Sum)
    return Account


def backTest(SignalList, PriceList, DayIn, PredictLength):
    # 分片
    SignalSheetList = []
    for i in range(PredictLength):
        SignalSheetList.append([])
    PriceSheetList = []
    for i in range(PredictLength):
        PriceSheetList.append([])

    Account = 0
    for i in range(len(SignalList)):
        j = i % PredictLength
        SignalSheetList[j].append(int(SignalList[i]))

    for i in range(len(PriceList)):
        j = i % PredictLength
        PriceSheetList[j].append(float(PriceList[i]))

    for i in range(PredictLength):
        Account = Account + backAccount(SignalSheetList[i], PriceSheetList[i], DayIn)

    Sum = DayIn * PredictLength
    Return_Rate = float(Account - Sum) / Sum

    return Return_Rate

def featureFit(list1,list2):
    list3 = []
    for i in range(len(list1)):
        for j in list2:
            if j == list1[i]:
                list3.append(i+1)
    return list3


# StockCodeList = createStockCodeList()
# PredictLength = 10

# for StockCode in StockCodeList:
#     filename = "/Users/gongxing/Desktop/shsz20Data/30data10/"+StockCode+".csv"
#     X_data, Y_data = loadDataSet(filename)
#
#     scaler = MinMaxScaler()
#     X_data = scaler.fit_transform(X_data)
#     Xtrain, Xtest, Ytrain, Ytest = TrainTestSplit(X_data, Y_data, 0.9)
#
#     # ctf = LogisticRegression(penalty='l2',solver='liblinear',C=0.5,max_iter=1000)
#     ctf = RandomForestClassifier(n_estimators=100,random_state=0)
#     ctf.fit(Xtrain, Xtest)
#     print(ctf.score(Ytrain, Ytest))
#     # print(accuracy_score(ctf.predict(Ytrain), Ytest))
#     SignalList = ctf.predict(Ytrain)
#     print(SignalList)

# PriceList = loadPriceList(filename, len(Ytest), PredictLength)
#
# print(backTest(SignalList, PriceList, 10000, PredictLength))

def LR(str_wl, str_prd, featureNum, testNum):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        newdata = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k=featureNum).fit_transform(data, label)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(newdata, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        newdata = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T,
                              k=featureNum).fit_transform(data, label)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(newdata, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf

def LRW(str_wl, str_prd, testNum):
    ww = []
    nw = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)
        # 划分数据集
        X_train, Y_train, X_test,Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train,Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        ww.append(acc)



        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nw.append(acc)

    return ww, nw
    # return ww, wf, nw, nf


def LR_RFE(str_wl, str_prd, featureNum, testNum):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)


        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf = RFE(ctf, featureNum)
        ctf.fit_transform(X_train, Y_train)
        # ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)


        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf = RFE(ctf, featureNum)
        ctf.fit_transform(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf


def LR_GBDT(str_wl, str_prd, testNum):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)
        data = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, label)


        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)
        data = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, label)

        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf

def LR_PCA(str_wl, str_prd, featureNum, testNum):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        data = PCA(n_components=featureNum).fit_transform(data)        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        data = PCA(n_components=featureNum).fit_transform(data)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf

def LR_LDA(str_wl, str_prd, testNum):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        data = LDA().fit_transform(data, label)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 赛选特征
        data = LDA().fit_transform(data, label)
        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf



def WGL(str_wl, str_prd, testNum, FeatureSelect, ML):
    ww = []
    nw = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)
        # 特征筛选
        if FeatureSelect == "GBDT":
            data = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, label)
        elif FeatureSelect == "LDA":
            data = LDA().fit_transform(data, label)
        elif FeatureSelect == "WITHOUT":
            data = data
        # 划分数据集
        X_train, Y_train, X_test,Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        if ML == 'LR':
            ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        elif ML == 'RF':
            ctf = RF(n_estimators=100, criterion='gini')
        elif ML == 'SVM':
            ctf = SVC(kernel='rbf', gamma=2)
        elif ML == 'NN':
            ctf = MLPClassifier(solver='sgd', activation='logistic')

        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        ww.append(acc)



        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)
        # 特征筛选
        if FeatureSelect == "GBDT":
            data = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, label)
        elif FeatureSelect == "LDA":
            data = LDA().fit_transform(data, label)
        elif FeatureSelect == "WITHOUT":
            data = data

        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        if ML == 'LR':
            ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        elif ML == 'RF':
            ctf = RF(n_estimators=100, criterion='gini')
        elif ML == 'SVM':
            ctf = SVC(kernel='rbf', gamma=2)
        elif ML == 'NN':
            ctf = MLPClassifier(solver='sgd',activation='logistic')

        ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nw.append(acc)

    return ww, nw

def FRP(str_wl, str_prd, featureNum, testNum, FeatureSelect, ML):
    wf = []
    nf = []
    StockCodeList = createStockCodeList()
    for StockCode in StockCodeList:
        filename = "/Users/gongxing/Desktop/ProcessData/network/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 特征选择
        if FeatureSelect == "FILTER":
            data = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k=featureNum).fit_transform(data, label)
        elif FeatureSelect == "PCA":
            data = PCA(n_components=featureNum).fit_transform(data)


        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        if ML == 'LR':
            ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        elif ML == 'RF':
            ctf = RF(n_estimators=100, criterion='gini')
        elif ML == 'SVM':
            ctf = SVC(kernel='rbf', gamma=2)
        elif ML == 'NN':
            ctf = MLPClassifier(solver='sgd',activation='logistic')

        if FeatureSelect == "RFE":
            ctf = RFE(ctf, featureNum)
            ctf.fit_transform(X_train, Y_train)
        else:
            ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        wf.append(acc)


        data, label = loadDataSet(filename, 1)
        # 标准化
        data = StandardScaler().fit_transform(data)

        # 特征选择
        if FeatureSelect == "FILTER":
            data = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T,
                               k=featureNum).fit_transform(data, label)
        elif FeatureSelect == "PCA":
            data = PCA(n_components=featureNum).fit_transform(data)

        # 划分数据集
        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, testNum)
        # 训练模型
        if ML == 'LR':
            ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        elif ML == 'RF':
            ctf = RF(n_estimators=100, criterion='gini')
        elif ML == 'SVM':
            ctf = SVC(kernel='rbf', gamma=2)
        elif ML == 'NN':
            ctf = MLPClassifier(solver='sgd', activation='logistic')


        if FeatureSelect == "RFE":
            ctf = RFE(ctf, featureNum)
            ctf.fit_transform(X_train, Y_train)
        else:
            ctf.fit(X_train, Y_train)
        # accuracy
        acc = ctf.score(X_test, Y_test)
        nf.append(acc)
    return wf, nf

def SaveWGL(week_list, day_list, FS_list,ML):
    for FS in FS_list:
        MAT = []
        for week in week_list:
            for day in day_list:
                wf_mean = []
                nf_mean = []

                wf, nf = WGL(week, day, 120, FS, ML)
                wf_mean.append(round(np.mean(wf), 4))
                nf_mean.append(round(np.mean(nf), 4))

                print(wf_mean)
                print(nf_mean)
                MAT.append(wf_mean)
                MAT.append(nf_mean)
        with open("/Users/gongxing/Desktop/result/"+ML+"/"+FS+".csv", 'w') as f:
            writer = csv.writer(f)
            # 将列表的每条数据依次写入csv文件， 并以逗号分隔
            # 传入的数据为列表中嵌套列表或元组，每一个列表或元组为每一行的数据
            writer.writerows(MAT)

def SaveFRP(week_list, day_list, FS_list, ML):
    for FS in FS_list:
        MAT = []
        for week in week_list:
            for day in day_list:
                wf_mean = []
                nf_mean = []

                for i in range(2, 12):
                    wf, nf = FRP(week, day, i, 120, FS, ML)
                    wf_mean.append(round(np.mean(wf), 4))
                    nf_mean.append(round(np.mean(nf), 4))

                print(wf_mean)
                print(nf_mean)
                MAT.append(wf_mean)
                MAT.append(nf_mean)
        with open("/Users/gongxing/Desktop/result/"+ML+"/"+FS+".csv", 'w') as f:
            writer = csv.writer(f)
            # 将列表的每条数据依次写入csv文件， 并以逗号分隔
            # 传入的数据为列表中嵌套列表或元组，每一个列表或元组为每一行的数据
            writer.writerows(MAT)

week_list = ['7','14','28','56']
day_list = ['5','10','30']
FSWGL = ["WITHOUT","GBDT","LDA"]
FSFRP = ["FILTER","PCA","RFE"]

SaveWGL(week_list, day_list, FSWGL,"NN")
SaveFRP(week_list, day_list, FSFRP,"NN")

