import time
import numpy as np
import pandas as pd
import operator
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# 加载数据
def loadData():
    """
    加载训练集和测试集数据

    Returns:
    train_data: 训练集特征数据
    label_data: 训练集标签数据
    test_data: 测试集特征数据
    """
    train = pd.read_csv('../cnn/train.csv')
    test = pd.read_csv('test.csv')

    train_data = train.values[:, 1:]  # 读入全部训练数据，返回numpy数组
    label_data = train.values[:, 0]  # 读入标签列（第0列）
    test_data = test.values[:, :]  # 读入全部测试数据
    return train_data, label_data, test_data


# 导出数据
def saveResult(result, filename):
    """
    导出预测结果

    Args:
    result: 预测结果
    filename: 导出文件名
    """
    print('开始导出数据...', end='')
    index = np.arange(len(result))
    newCsv = pd.DataFrame({'imageId': index, 'label': result})
    newCsv.to_csv(filename, index=False)
    print('导出完毕!')


# KNN算法
def KNN(in_x, x_labels, y_labels, k):
    """
    K最近邻算法实现

    Args:
    in_x: 待预测样本特征
    x_labels: 训练集特征数据
    y_labels: 训练集标签数据
    k: K值

    Returns:
    预测结果标签
    """
    x_labels_size = x_labels.shape[0]
    distance = (np.tile(in_x, (x_labels_size, 1)) - x_labels) ** 2
    ad_distance = distance.sum(axis=1)
    sq_distance = ad_distance ** 0.5
    ed_distance = sq_distance.argsort()
    classdict = {}
    for i in range(k):
        vote_label = y_labels[ed_distance[i]]
        classdict[vote_label] = classdict.get(vote_label, 0) + 1
    sort_classdict = sorted(classdict.items(), key=operator.itemgetter(1), reverse=True)
    return sort_classdict[0][0]


# 计算损失
def calculateLoss(test_data, test_labels, X_train, y_train, k):
    count = 0
    total = len(test_data)
    for i in tqdm(range(total)):
        if KNN(test_data[i], X_train, y_train, k) != test_labels[i]:
            count += 1
    loss = count / total
    return loss


# 执行主函数
if __name__ == '__main__':
    startTime = time.perf_counter()  # 定位程序开始时间
    print('开始执行...')

    trainData, labelData, testData = loadData()  # 加载数据
    X_train, X_test, y_train, y_test = train_test_split(trainData, labelData, test_size=0.2,
                                                        random_state=12)  # 数据分类，一部分做测试集

    epochs = 5
    k = 5  # KNN中的K值
    for epoch in range(1, epochs + 1):
        # 计算并输出损失
        train_loss = calculateLoss(X_test, y_test, X_train, y_train, k)
        print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}')

        # 更新KNN模型参数
        # 这里可以添加训练过程中的其他操作

    test_labels = []
    for i in tqdm(range(len(testData))):
        test_labels.append(KNN(testData[i], X_train, y_train, 3))

    saveResult(test_labels, 'result.csv')  # 导出数据

    endTime = time.perf_counter()  # 定位程序结束时间
    costTime = endTime - startTime  # 计算总耗时
    print('执行完毕，总耗时：{:.0f}秒'.format(costTime))
