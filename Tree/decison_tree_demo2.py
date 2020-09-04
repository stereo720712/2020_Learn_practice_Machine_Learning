from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math
import  pandas as pd
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


datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
# _, y_train, features = data_df.iloc[:, :-1], data_df.iloc[:, -1], data_df.columns[: -1]

X_train = data_df.iloc[:,:4]
y_traget = data_df.iloc[:,-1:]


tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_traget)
