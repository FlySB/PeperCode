import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# 创建股票波动率矩阵(用于计算相关系数矩阵)和股票代码列表(用于画图)
def createVolatilityMat():
    StockCodeList = []
    VolatilityMat = []
    with open("/Users/gongxing/Desktop/data/shsz20.csv",encoding="gb2312") as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)
        i = 0
        for line in reader:
            StockCode = str(line[0])
            Id_StockCode = (i, StockCode)
            StockCodeList.append(Id_StockCode)
            i = i+1
            with open("/Users/gongxing/Desktop/StockData/" + StockCode + ".csv", encoding="gb2312") as csvfile_1:
                VolatilityList = []
                reader_1 = csv.reader(csvfile_1)
                next(reader_1)
                for line in reader_1:
                    VolatilityList.append(round(float(line[1]), 2))
                # time_lenth = len(VolatilityList)
                # if time_lenth != 443:
                #     print(StockCode)
                VolatilityMat.append(VolatilityList)
        return VolatilityMat,StockCodeList

# 建立邻接矩阵
# MAT: 波动率矩阵
def createGraphMat(MAT):
    GraphMat = []
    CollectionList = []
    PersonMat = np.corrcoef(MAT)


    for line in PersonMat:
        CollectionList.append(sum(line)-1)
        G_list = []
        for i in line:
            if i >= 0.55 and i < 0.999:
                i = 1
            else:
                i = 0
            G_list.append(i)
        GraphMat.append(G_list)
    GraphMat = np.array(GraphMat)
    return GraphMat,CollectionList


# 绘制邻接矩阵矩阵图
# GraphMat:邻接矩阵
# StockCodeList:股票代码列表
def drawGraph(GraphMat, StockCodeList):

    # 邻接矩阵转换成图
    graph = nx.from_numpy_matrix(GraphMat)
    # 画图
    labels = dict(StockCodeList)
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, labels=labels, node_color='pink', node_size=200)
    plt.show()

# 建立窗口矩阵列表
# MAT:全体波动率矩阵
# TimeLength:全体数据的时间跨度，既MAT行长度
# WindowLength:窗口大小
def createWindowMatList(MAT,TimeLength, WindowLength):
    MAT_list = []
    for i in range(TimeLength - WindowLength + 1):
        small_mat = []
        for l in MAT:
            small_list = l[i:i+WindowLength]
            small_list = np.array(small_list)
            small_mat.append(small_list)
        MAT_list.append(small_mat)
    return MAT_list

# 建立平均聚集系数列表
# MatList:窗口矩阵列表
def createClusterlist(MatList):
    ClusterList = []
    for mat in MatList:
        GraphMat, CollectionList = createGraphMat(mat)
        graph = nx.from_numpy_matrix(GraphMat)

        # 聚集系数
        average_clustering = nx.average_clustering(graph)
        average_clustering = round(average_clustering, 2)
        ClusterList.append(average_clustering)
    return ClusterList

# 建立点聚集系数列表
# MatList:窗口矩阵列表
def createPointClusterlist(MatList, StockNum):
    ClusterList = []
    for i in range(StockNum):
        ClusterList.append([])
    DegreeList = []
    for i in range(StockNum):
        DegreeList.append([])

    CloseList = []
    for i in range(StockNum):
        CloseList.append([])

    BetweennessList = []
    for i in range(StockNum):
        BetweennessList.append([])

    CollectionList = []
    for i in range(StockNum):
        CollectionList.append([])


    for mat in MatList:
        GraphMat,_collectionList = createGraphMat(mat)
        graph = nx.from_numpy_matrix(GraphMat)

        _closeList = nx.closeness_centrality(graph)
        _closeList = list(_closeList.values())

        _btList = nx.betweenness_centrality(graph)
        _btList = list(_btList.values())

        # 聚集系数
        for i in range(StockNum):
            ClusterList[i].append(round(nx.clustering(graph,i),4))
            DegreeList[i].append(round(nx.degree(graph, i)/(StockNum-1), 4))
            CloseList[i].append(round(_closeList[i], 4))
            BetweennessList[i].append(round(_btList[i], 4))
            CollectionList[i].append(round(_collectionList[i], 4))
    return DegreeList, ClusterList, BetweennessList, CloseList, CollectionList


# 绘制聚集系数折线图
# ClusterList:聚集系数列表
def drawClusterlist(Clusterlist,TimeLength, WindowLength):
    date_list = []
    with open("/Users/gongxing/Desktop/data/sh.600000.csv", encoding="gb2312") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            date_list.append(line[0])
    x_date_list = date_list[0:TimeLength - WindowLength + 1]

    # 设置画布大小
    plt.figure(figsize=(16, 4))
    # 标题
    plt.title("clustering")
    # 数据
    plt.plot(x_date_list, Clusterlist, label='average_clustering', linewidth=3, color='r', marker='o',
             markerfacecolor='blue', markersize=8)

    # 横坐标描述
    plt.xlabel('month')
    # 纵坐标描述
    plt.ylabel('clustering')

    # 设置数字标签
    for a, b in zip(x_date_list, Clusterlist):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=20)

    plt.legend()
    plt.show()

# 涨跌标记
def UpDownMark(StockCodeList, PredictLength):
    PriceMat = []
    for StockCode in StockCodeList:
        StockCode = StockCode[1]
        PriceList = []
        with open("/Users/gongxing/Desktop/StockData/" + StockCode + ".csv", encoding="gb2312") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                PriceList.append(float(line[2]))
        PriceMat.append(PriceList)

    MarksMat = []
    for pricelist in PriceMat:
        MarksList = []
        for i in range(len(pricelist)-PredictLength):
            temp = pricelist[i+PredictLength]-pricelist[i]
            if temp > 0:
                MarksList.append(1)
            else:
                MarksList.append(0)
        MarksMat.append(MarksList)

    return MarksMat

# 数据处理储存
def DateProcess(DegreeList, ClusterList, BetweennessList, CollectionList ,CloseList, MarksMat, StockCodeList, WindowLength):
    mat = []
    for StockCode in StockCodeList:
        StockCode = StockCode[1]
        list = []
        with open("/Users/gongxing/Desktop/StockData/" + StockCode + ".csv", encoding="gb2312") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                if line[8] == '':
                    line[8] = '0'
                list.append(line)
        mat.append(list)

    TimeLength = len(mat[0])

    MAT = []
    # for i in range(len(mat)):
    #     list = mat[i][-(TimeLength-WindowLength+1):]
    #
    #     for j in range(len(list)):
    #         list[j].append(DegreeList[i][j])
    #         list[j].append(ClusterList[i][j])
    #         list[j].append(BetweennessList[i][j])
    #         c = float(CloseList[i][j])
    #         if math.isnan(c):
    #             list[j].append(0)
    #         else:
    #             list[j].append(c)
    #         list[j].append(CollectionList[i][j])
    #     MAT.append(list)

    for i in range(len(MAT)):
        for j in range(len(MarksMat[0])-WindowLength+1):
            MAT[i][j].append(MarksMat[i][j+WindowLength-1])

    for i in range(len(StockCodeList)):
        with open("/Users/gongxing/Desktop/ProcessData/without/7data5/" + StockCodeList[i][1] + ".csv", 'w') as f:
            writer = csv.writer(f)
            # 将列表的每条数据依次写入csv文件， 并以逗号分隔
            # 传入的数据为列表中嵌套列表或元组，每一个列表或元组为每一行的数据
            writer.writerows(MAT[i])

    return MAT

# 数据处理储存
def DateProcess2(MarksMat, StockCodeList, WindowLength):
    mat = []
    for StockCode in StockCodeList:
        StockCode = StockCode[1]
        list = []
        with open("/Users/gongxing/Desktop/StockData/" + StockCode + ".csv", encoding="gb2312") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                if line[8] == '':
                    line[8] = '0'
                list.append(line)
        mat.append(list)

    TimeLength = len(mat[0])

    MAT = []
    for i in range(len(mat)):
        list = mat[i][-(TimeLength-WindowLength+1):]
        MAT.append(list)


    for i in range(len(MAT)):
        for j in range(len(MarksMat[0])-WindowLength+1):
            MAT[i][j].append(MarksMat[i][j+WindowLength-1])

    for i in range(len(StockCodeList)):
        with open("/Users/gongxing/Desktop/ProcessData/without/7data/" + StockCodeList[i][1] + ".csv", 'w') as f:
            writer = csv.writer(f)
            # 将列表的每条数据依次写入csv文件， 并以逗号分隔
            # 传入的数据为列表中嵌套列表或元组，每一个列表或元组为每一行的数据
            writer.writerows(MAT[i])

    return MAT



windowlength = 7
predictlength = 10

VolatilityMat,StockCodeList = createVolatilityMat()
print(StockCodeList)

MAT_list = createWindowMatList(VolatilityMat, len(VolatilityMat[0]), windowlength)

MarksMat = UpDownMark(StockCodeList,predictlength)
MAT = DateProcess2(MarksMat, StockCodeList, windowlength)









