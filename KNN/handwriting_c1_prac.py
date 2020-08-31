## KNN example ,利用KNN做手寫數字辨別
import  numpy as np
from  os import  listdir
from collections import Counter

def img2vector(filename):
    """
    將圖像數據轉換為向量
    :param filename: 圖片文件,圖片格式為 32*32
    :return: one dim array
    function :
    將圖像轉換為向量：該函數創建 1*1024的numpy array, 打開給定的文件
    循環讀出前32行,並將美航的頭32個字符存在數組中,最後返回
    32*32 --> 1*1024
    """
    return_vector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(lineStr[j])
    return  return_vector

def classify(inX, dataSet, labels, k):
    """

    :param inX:[1,2,3] , 用於分類的輸入量
    :param dataSet:[[1,2,3],[1,2,0]]  輸入的訓練樣本集
    :param labels: 標籤向量
    :param k: 選擇最近的鄰居數目
    attention: labels元素數目和dataSet 行數相同,這邊用歐式距離判斷遠近
    :return:
    """
    dataSetSize = dataSet.shape[0]
    # https://ppt.cc/fD7ovx
    inX_expand = np.tile(inX,(dataSetSize,1))

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # distance
    # get square
    sqDiffMat = diffMat ** 2
    # add every row in matrix
    sq_Distance = sqDiffMat.sum(axis=1)
    # 開更號
    distances = sq_Distance ** 0.5
    # 根據距離排序從小到大的排序 , 返回對應的索引位置
    # argsort() 是將x中的元素從小到大排列 提取其對應的index,輸出y
    # ex: y = array([3,0,2,-1,4,9]) , x[3] = -1 最小, x[5] = 9 最大 所以y[5] = 5
    # print('distances=' , distances)
    # https://ppt.cc/fB2Bbx
    sorted_dist_indicies = distances.argsort() # distance sort 
    # print('distances.argsort() = ', sorted_dist_indicies);

    # 2, 選擇距離最小的k個點
    classCount = {}
    for i in range(k):
        # 找到該樣本類型
        voteI_label = labels[sorted_dist_indicies[i]]
        classCount[voteI_label] = classCount.get(voteI_label, 0) + 1
    # 3. 排序並找出出現最多的那個類型
    maxClassCount = max(classCount, key=classCount.get)
    return  maxClassCount




def hand_writing_class_test():
    # 1.導入數據
    hw_labels = []
    training_file_list = listdir('trainingDigits') # load the training set
    file_list_len = len(training_file_list) # file number
    training_Mat = np.zeros((file_list_len,1024))

    # hw_labels 存儲 0 ~ 9 對應的index 位置 , trainingMat 存放每個位置對應的圖片向量
    for i in range(file_list_len):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0] # take off .txt
        class_num_str = int(file_str.split('_')[0]) # which number
        hw_labels.append(class_num_str)
        # 將 32*32 matrix ==> 1*1024
        training_Mat[i, :] = img2vector('trainingDigits/%s' % file_name_str) # training data

    # 2.導入測試數據
    test_file_list = listdir('testDigits')  # iterate test set
    errorCount = 0
    len_test_file_list = len(test_file_list)

    for i in range(len_test_file_list):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_Mat,hw_labels, 1)
        print("the classifier came back with: %s, the real answer is: %s" % (classifier_result, class_num_str))

        if (classifier_result != class_num_str) : errorCount += 1.0

    print("\n the total number of errors is : %d" % errorCount)
    print("\n the total error rate is %f " % (errorCount/float(file_list_len)))


if __name__ == '__main__':
    hand_writing_class_test()

