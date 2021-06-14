from paper_sklearn import loadDataSet,TrainTestSplit,createStockCodeList
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np

StockCodeList = createStockCodeList()
print(StockCodeList)
def createLabelMat(StockCodeList,str_wl, str_prd):
    preMAT = []
    priceMAT = []
    for StockCode in StockCodeList:
        filename = "data/ProcessData/"+str_wl+"data"+str_prd+"/" + StockCode + ".csv"

        data, label = loadDataSet(filename,0)
        prelist = loadPriceList(filename,120,int(str_prd))
        priceMAT.append(np.array(prelist))
        # 标准化
        data = StandardScaler().fit_transform(data)

        X_train, Y_train, X_test, Y_test = TrainTestSplit(data, label, 120)

        ctf = LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
        ctf.fit(X_train,Y_train)
        prelist = ctf.predict(X_test)
        preMAT.append(prelist)
    return preMAT,priceMAT

def loadPriceList(fileName, YLength, PredictLength):
    PriceList = []
    with open(fileName, encoding="gb2312") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            PriceList.append(float(line[2]))

    return PriceList[-(YLength+PredictLength):]

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

def backTestAll(stockList,preMAT, priceMAT,DayMoney, preLength):
    rate = 0
    for i in range(len(stockList)):
        rate = rate + backTest(preMAT[i],priceMAT[i],DayMoney,preLength)
    return rate/len(stockList)

MAT1, MAT2 = createLabelMat(StockCodeList,'7','5')
print(MAT1)
print(MAT2)
rate = backTestAll(StockCodeList,MAT1,MAT2,1000000,5)
print(rate)