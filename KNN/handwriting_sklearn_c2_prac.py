# use sklearn KneighborsClassifier to determine hand writing number from c1

from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np
from  os import listdir

def img2vector(file_name):
    '''
    img : 32 x 32 to 1x1024

    :param file_name:
    :return: one dim array
    '''
    vector = np.zeros((1,1024))
    fr = open(file_name)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(lineStr[j])
    return vector


if __name__ == '__main__':
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    file_list_len = len(training_file_list)
    hw_data = np.zeros((file_list_len,1024))

    # get labels
    for i in range(file_list_len):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0] # remove .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        hw_data[i, :] = img2vector('trainingDigits/%s' % file_name_str)
        #embedding data

    # https://ppt.cc/ffXjXx
    train_data, test_data, train_label, test_label = train_test_split(hw_data, hw_labels,test_size=0.2)
    knn = KNeighborsClassifier()
    knn.fit(train_data,train_label)
    res = knn.predict(test_data)
    score = knn.score(test_data, test_label)
    print(score)



    pass
