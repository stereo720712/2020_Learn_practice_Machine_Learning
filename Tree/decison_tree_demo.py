import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log


def create_data():
    '''

    :return:  demo dataset

    '''
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]

    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels


def calc_ent(datasets):
    data_length = len(datasets)
    labels_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in labels_count:
            labels_count[label] = 0
        labels_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2)
                for p in labels_count.values()])
    return ent


# condition ent , for all condition
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]  # index
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    # print(feature_sets)
    cond_ent = sum(
        [(len(p) / data_length) * calc_ent(p) for p in feature_sets.values()])
    temp_cont_ent = 0
    for p in feature_sets.values():
        temp = (len(p) / data_length) * calc_ent(p)
        temp_cont_ent = temp_cont_ent + temp

    return cond_ent


# info gain
def info_gain(ent, cond_ent):
    return ent - cond_ent


def info_gain_train(datasets):
    count = len(datasets[0]) - 1  # 特徵數
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        # [[年龄，a],[有房子，b]]
        print('特徵({}) - info_gain = {:.3f}'.format(labels[c], c_info_gain))
    best_ = max(best_feature, key=lambda x: x[-1])
    return '特徵({})的信息增益最大， ˊˇ選擇為跟節點特徵'.format(labels[best_[0]])


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        '''
        1.葉子節點 label
        2.中間節點 條件:[條件1];[桃件2]
        '''
        self.root = root  # 是否為root
        self.label = label  # 葉節點所有樣本標籤
        self.feature_name = feature_name  # 切分條件
        self.feature = feature
        self.tree = {}  # son node
        self.result = {
            'label': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        '''traing  use'''
        self.tree[val] = node

    def predict(self, features):
        '''
          features => 要預測的數據
      '''
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

# tree root node
class Dtree:
    '''
        建樹過程
    '''
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    # ent
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1] # tag
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = sum([(p / data_length) * log(p / data_length, 2)
                   for p in label_count.values()])
        return ent

    # condition ent
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p)/data_length) * self.calc_ent(p)
                        for p in feature_sets.values()])
        return cond_ent

    # ent gain
    @staticmethod
    def info_gain(ent, cond_ent):
        return  ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []

        # for all feature
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))

        # compare
        best_ = max(best_feature, key=lambda x:x[-1])
        return best_

    def train(self, train_data):
        '''
        ch2
        input: dataset(DataFrame),feature set A, limit epc
        output: tree
        :param train_data:
        :return:
        '''

        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[: -1]

        # 1.若D中實例屬於同一類Ck, 則T為單節點樹,並將類Ck作爲節點的類標記, 返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2.若A為空,則T為單節點樹, 並將類Ck作為結點的類標記,返回T
        if len(features == 0):
            return Node(root=True,
                        label=y_train.value_counts().sort_values(
                ascending=False).index[0])

        # 3.計算最大信息增益 同5.1 Ag為信息增益最大的特徵
        max_feature, max_info_gain = self.info_gain_train(np.array((train_data)))
        max_feature_name = features[max_feature]

        # 4.Ag的信息增益小於閥值epc, 則置T為單節點樹,並將D中實例數最大的類Ck作為該節點的標記
        if max_info_gain < self.epsilon:
            return Node(
                root=True,
                label=y_train.value_counts.sort_values(ascending=False).index[0]
            )

        #5.構建子集
        node_tree = Node(
             root=False,
             feature_name=max_feature_name,
             feature=max_feature
        )
        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] ==
                                          f].drop([max_feature_name], axis=1)
            # 6. recurive generate tree
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f,sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def prediect(self, X_test):
        return self._tree.predict(X_test)












datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = Dtree()
tree = dt.fit(data_df)

print(tree)

print(dt.prediect(['老年', '否', '否', '一般']))


# print(train_data)
# print(labels)
# cond_ent(datasets)
# print(info_gain_train(datasets))
