'''
https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification/notebooks
Text Classification with ML(NB,LR,RF)
'''

import nltk
## http://www.nltk.org/book/

from nltk.corpus import stopwords
## https://webcache.googleusercontent.com/search?q=cache:G4dy-_IN1XUJ:https://medium.com/%40sfhsu29/nlp-%25E5%2585%25A5%25E9%2596%2580-1-2-stop-words-da3d311d29bc+&cd=2&hl=zh-TW&ct=clnk&gl=tw&client=firefox-b-d

# nltk.download('stopwords')

print(stopwords.words('english'))

'''
Data preparation
Feature Engineering
    - Count vector as feature
    - Tf-idf as feature #===========?
    - word embedding as feature
    - Text or NLP based Feature #==========?
    - Topic model as feature

Training all type of model

Evaluating accuracy

confusion matrix to check accuracy #============?
#fuzzy data to verify ?

    
'''

# 1. Data Preparation
import pandas as pd

data = pd.read_csv("../bbc_news_Data/ans1/bbc-text.csv")

print(data.head())
#print(data.info)

data['label'] = data['category']
data['input'] = data['text']
data.drop(['category','text'], axis=1, inplace=True)

data['label'].unique()
print(data['label'].unique())

# label encoding for labels
from sklearn.preprocessing import  LabelEncoder

encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])
print(data)

# https://www.itread01.com/content/1544695933.html
# https://ithelp.ithome.com.tw/articles/10228481?sc=iThelpR
# count , tf-idf vector
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
from  sklearn.model_selection import  train_test_split

#count vector
count_vect = CountVectorizer(analyzer='word')
count_vect_X = count_vect.fit_transform(data["input"])
cvtrain_x, cvtest_x, cvtrain_y, cvtest_y = train_test_split(
    count_vect_X, data['label'], test_size=0.2)

# tfidf vector for text on word level
tfidf_obj = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf_vec_X = tfidf_obj.fit_transform(data['input'])
tfidf_train_x, tfidf_test_x, tfidf_train_y, tfidf_test_y = train_test_split(tfidf_vec_X,data['label'], test_size=0.2)

# some ref
# https://medium.com/machine-learning-intuition/document-classification-part-2-text-processing-eaa26d16c719
# tfidf on ngram level #?????????

tfidf_ngram = TfidfVectorizer(analyzer='word',ngram_range=(2, 3), max_features=5000)
tfidf_ngram_X = tfidf_ngram.fit_transform(data['input'])
ngram_train_x, ngram_test_x, ngram_train_y, ngram_test_y = train_test_split(
    tfidf_vec_X, data['label'], test_size=0.2)

from sklearn import metrics
#utility function for model building
def train_model(model_classifier,  train_x, test_x, train_y, test_y):
    model_classifier.fit(train_x, train_y)
    prediction = model_classifier.predict(test_x)
    print(prediction)
    return metrics.accuracy_score(prediction,test_y)

# Naive bayes Classifer
from  sklearn import  linear_model, naive_bayes, metrics, svm

# naive bayes model on count_vector
# https://blog.csdn.net/TeFuirnever/article/details/100125386
accuracy = train_model(naive_bayes.MultinomialNB(), cvtrain_x, cvtest_x,
                       cvtrain_y, cvtest_y)

print('NB, Count Vecotrs: ', accuracy)

# naivebayes model on tfidf vector
accuracy = train_model(naive_bayes.MultinomialNB(),
                       tfidf_train_x, tfidf_test_x, tfidf_train_y, tfidf_test_y)
print('NB, Tfidf word level Vectors:', accuracy)

# naivebayes model on tfidf ngram vector
accuracy = train_model(naive_bayes.MultinomialNB(), ngram_train_x,
                       ngram_test_x, ngram_train_y, ngram_test_y)
print('NB Tfidf ngram level Vectors:', accuracy)



# Logistic Regression
accuracy = train_model(linear_model.LogisticRegression(), cvtrain_x, cvtest_x,
                       cvtrain_y, cvtest_y)
print('Logistic regression , count Vectors:', accuracy)

# Logistic Regression model on tfidf vector
accuracy = train_model(linear_model.LogisticRegression(),
                       tfidf_train_x, tfidf_test_x, tfidf_train_y, tfidf_test_y)

# Logistic Regression model on tfidf ngram vector
accuracy = train_model(linear_model.LogisticRegression(), ngram_train_x, ngram_test_x,
                       ngram_train_y, ngram_test_y)
print('Logistic Regression, Tfidf ngram Vectors:', accuracy)


# support Vector machine on count vector
accuracy = train_model(svm.SVC(), cvtrain_x, cvtest_x,
                       cvtrain_y, cvtest_y)
print('Support vector machine , count Vectors:', accuracy)
# support Vector machine on tfidf vector
accuracy = train_model(svm.SVC(), tfidf_train_x, tfidf_test_x,
                       tfidf_train_y, tfidf_test_y)
print("Support vector machine,Tfidf word level Vectors: ", accuracy)
# support Vector machine on tfidf vector
accuracy = train_model(svm.SVC(),ngram_train_x,ngram_test_x,ngram_train_y,ngram_test_y)
print("Support vector machine,Tfidf ngram Vectors: ", accuracy)

# Bagging/ Ensemble model - Random Forest

from sklearn import ensemble

# random forest on count vector
accuracy = train_model(ensemble.RandomForestClassifier(),cvtrain_x,cvtest_x,cvtrain_y,cvtest_y)
print("Random forest classifier,count Vectors: ", accuracy)

# random forest on tfidf word vector
accuracy = train_model(ensemble.RandomForestClassifier(),tfidf_train_x,tfidf_test_x,tfidf_train_y,tfidf_test_y)
print("Random forest classifier,Tfidf word level Vectors: ", accuracy)

# random forest on tfidf ngram vector
accuracy = train_model(ensemble.RandomForestClassifier(),ngram_train_x,ngram_test_x,ngram_train_y,ngram_test_y)
print("Random forest classifier,Tfidf ngram Vectors: ", accuracy)
