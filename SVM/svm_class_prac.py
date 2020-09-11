'''
libsvm库安装

LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM模式识别与回归的软件包。其它的svm库也有，这里以libsvm为例。

libsvm下载地址：
http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip

MAC系统

1.下载libsvm后解压，进入目录有个文件：libsvm.so.2，把这个文件复制到python安装目录site-packages/下。

2.在site-packages/下新建libsvm文件夹，并进入libsvm目录新建init.py的空文件。

3.进入libsvm解压路径：libsvm-3.23/python/，把里面的三个文件：svm.py、svmutil.py、commonutil.py，复制到新建的：site-packages/libsvm/目录下。之后就可以使用libsvm了。
Windows系统

安装教程：https://www.cnblogs.com/bbn0111/p/8318629.html


'''

import sys
import os
import jieba
from libsvm import svm
from libsvm.svmutil import svm_train, svm_predict, svm_save_model, svm_load_model
from libsvm.commonutil import svm_read_problem

news_file='cnews.train.txt'
test_file='cnews.test.txt'

output_word_file='cnews_dict.txt'
output_word_test_file='cnews_dict_test.txt'

# 最後生成的詞向量文件
feature_file = 'cnews_feature_file.txt'
feature_test_file = 'cnews_feature_test_file.txt'
# 模型最後保存文件
model_filename='cnews_model'

# multi hot
'''
对于某个属性对应的分类特征,可能该特征下有多个取值,比如一个特征表示对哪些物品感兴趣,那么这个特征不是单个值,而是有多个取值,样本1 
在该属性下取值有1,2两种特征,  样本2 在该属性下有2一种特征, 样本3 在该属性下有3,4 两种特征,
如果以类似one-hot编码的形式来定义特征应为样本1 [1,1,0,0]  样本2 [0,1,0,0], 样本3 [0,0,1,1],
但是这种变量不能够直接用embedding_lookup去做,
embedding_lookup只接受只有一个1的one-hot编码,那么为了完成这种情况的embedding需要两个步骤:

1. 将输入属性转化为类型one-hot编码的形式, 在tensorflow中这个过程是通过tf.SparseTensor来完成的,实际上就是构建了一个字典矩阵,
key为坐标,value为1或者0表示是否有值,对于一个样本如样本1来说就是构建了一个矩阵[[1,1,0,0]]表示有物品1和2,
这个矩阵的规模为[batch_size,num_items],这个过程即为multi-hot编码

2. 将构建好的类似于one-hot编码的矩阵与embedding矩阵相乘, 
embedding矩阵的规模为[num_items, embedding_size],相乘之后得到的输出规模为[batchi_size, embedding_size],
即对多维多属性的特征构建了embedding vector

'''

with open(news_file, 'r') as f:
    lines = f.readlines()

label, content = lines[0].strip('\r\n').split('\t')
print(content)
words_iter = jieba.cut(content)     # 分詞
print('/'.join(words_iter))


def generate_word_file(input_char_file, output_word_file):
    with open(input_char_file, 'r') as f:
        lines = f.readlines()

    with open(output_word_file, 'w') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            words_iter = jieba.cut(content)
            word_content = '';
            for word in words_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line)

#generate_word_file(news_file, output_word_file)
#generate_word_file(test_file, output_word_test_file)
print('==============分詞完成================')

class Category:  # 分類 topic
    def __init__(self, category_file):
        self._category_to_id = {}
        with open(category_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            category, idx = line.strip('\r\n').split('\t')
            idx = int(idx)
            self._category_to_id[category] = idx

    def category_to_id(self, category):
        return self._category_to_id[category]

    def size(self):
        return len(self._category_to_id)


category_file='cnews.category.txt'
category_vocab=Category(category_file)
print(category_vocab.size())

#所有文檔原始分詞的集合併集 -> 詞表大小、

def generate_feature_dict(train_file, feature_threshold=10):
    feature_dict = {}
    with open(train_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split(' '):
            if not word in feature_dict:
                feature_dict.setdefault(word, 0)
            feature_dict[word] += 1

    filtered_feature_dict = {}
    for feature_name in feature_dict:
        if feature_dict[feature_name] < feature_threshold:
            continue
        if not feature_name in filtered_feature_dict:
            filtered_feature_dict[feature_name] = len(filtered_feature_dict) + 1

    return filtered_feature_dict

feature_dict = generate_feature_dict(output_word_file, feature_threshold=100)
print(len(feature_dict))


# word vector
def generate_feature_line(line, feature_dict, category_vocab):
    label, content = line.strip('\r\n').split('\t')
    label_id = category_vocab.category_to_id(label)
    feature_example = {}
    for word in content.split(' '):
        if not word in feature_dict:
            continue
        feature_id = feature_dict[word]
        feature_example.setdefault(feature_id,0)
        feature_example[feature_id]+=1
    feature_line = '%d' % label_id
    sorted_feature_example = sorted(feature_example.items(), key=lambda d:d[0])
    for item in sorted_feature_example:
        feature_line += '%d:%d' % item
    return feature_line


# loop to get vector doc

def convent_raw_to_feature(raw_file, feature_file, feature_dict, category_vocab):
    with open(raw_file, 'r') as f:
        lines = f.readlines()
    with open(feature_file, 'w') as f:
        for line in lines:
            feature_line = generate_feature_line(line, feature_dict,category_vocab)
            f.write('%s\n' % feature_line)

## test data use same word table
#convent_raw_to_feature(output_word_file, feature_file, feature_dict, category_vocab)
#convent_raw_to_feature(output_word_test_file, feature_test_file, feature_dict, category_vocab)
print('====== accomplish word vector============')

# generate svn train data
train_label, train_value =svm_read_problem(feature_file)
print(train_label[0], train_value[0])

train_test_label, train_test_value = svm_read_problem(feature_test_file)

if (os.path.exists(model_filename)):
    model = svm_load_model(model_filename)
else:
    model=svm_train(train_label,train_value,'-s 0 -c 5 -t 0 -g 0.5 -e 0.1')
    svm_save_model(model_filename,model)

p_labs, p_acc, p_vals = svm_predict(train_test_label, train_test_value, model)
print(p_acc)

