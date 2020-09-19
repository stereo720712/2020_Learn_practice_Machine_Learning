# https://www.kaggle.com/gauravsb/text-classification-in-machine-learning

import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model, naive_bayes, metrics, svm

CATEGORY = 'category'
LABEL = 'label'
INPUT = 'input'
TEXT = 'text'


print(stopwords.words('english'))


'''
Data preparation

Feature Engineering
    - Count vector as feature
    - Tf-idf as feature
    - word embedding as feature
    - Text or NLP base feature
    - Topic model as feature
Training all type of model
Evaluating accuracy
confusion matrix to check the accuracy

'''

# 1. Data preparation
data = pd.read_csv("./bbc-text.csv")
print(data.head())
print(data.info())

data[LABEL] = data[CATEGORY]
data[INPUT] = data[TEXT]
data.drop([CATEGORY, TEXT], axis=1, inplace=True)
print(data[LABEL].unique())
print(data.head(3))

# label encoding for labels

label_encoder = LabelEncoder()
data[LABEL] = label_encoder.fit_transform(data[LABEL])

# count vector for text
count_vect =CountVectorizer(analyzer='word')
count_vect_X = count_vect.fit_transform(data[INPUT])
cvtrain_x, cvtest_x, cvtrain_y,cvtest_y = train_test_split(count_vect_X, data[LABEL], test_size=0.2)

# tfidf vector for text on word level
tfidf_obj = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf_vec_X = tfidf_obj.fit_transform(data[INPUT])
tfidf_train_x, tfidf_test_x, tfidf_train_y, tfidf_test_y = train_test_split(tfidf_vec_X,data[LABEL],test_size=0.2)

# tfidf on ngram level
tfidf_ngram = TfidfVectorizer(analyzer='word', ngram_range=(1,3),max_features=5000)
tfidf_vec_X = tfidf_ngram.fit_transform(data[INPUT])
ngram_train_x, ngram_test_x, ngram_train_y, ngram_test_y = train_test_split(tfidf_vec_X, data[LABEL],test_size=0.2)

def train_model(model_calssifier, train_x, test_x, train_y, test_y):
    model_calssifier.fit(train_x, train_y)
    # save model
    # pass
    prediction = model_calssifier.predict(test_x)
    print(prediction)
    return metrics.accuracy_score(prediction, test_y)

# Naive_bayes Classifier
accuracy = train_model(naive_bayes.MultinomialNB(),cvtrain_x,cvtest_x,cvtrain_y,cvtest_y)
print('NB, Count Vectors:', accuracy)

# copy paste
# naivebayes model on tfidf vector
accuracy = train_model(naive_bayes.MultinomialNB(),tfidf_train_x,tfidf_test_x,tfidf_train_y,tfidf_test_y)
print("NB,Tfidf word level Vectors: ", accuracy)


# naivebayes model on tfidf ngram vector
accuracy = train_model(naive_bayes.MultinomialNB(),ngram_train_x,ngram_test_x,ngram_train_y,ngram_test_y)
print("NB,Tfidf ngram level Vectors: ", accuracy)


# Logistic Regression

accuracy = train_model(linear_model.LogisticRegression(),cvtrain_x,cvtest_x,cvtrain_y,cvtest_y)
print("Logistic regression,count Vectors: ", accuracy)

# Logistic Regression model on tfidf vector
accuracy = train_model(linear_model.LogisticRegression(),tfidf_train_x,tfidf_test_x,tfidf_train_y,tfidf_test_y)
print("Logistic Regression,Tfidf word level Vectors: ", accuracy)


# Logistic Regression model on tfidf ngram vector
accuracy = train_model(linear_model.LogisticRegression(),ngram_train_x,ngram_test_x,ngram_train_y,ngram_test_y)
print("Logistic Regression,Tfidf ngram Vectors: ", accuracy)


# support Vector machine on count vector
accuracy = train_model(svm.SVC(),cvtrain_x,cvtest_x,cvtrain_y,cvtest_y)
print("Support vector machine,count Vectors: ", accuracy)

# support Vector machine on tfidf vector
accuracy = train_model(svm.SVC(),tfidf_train_x,tfidf_test_x,tfidf_train_y,tfidf_test_y)
print("Support vector machine,Tfidf word level Vectors: ", accuracy)


# support Vector machine on tfidf ngram vector
accuracy = train_model(svm.SVC(),ngram_train_x,ngram_test_x,ngram_train_y,ngram_test_y)
print("Support vector machine,Tfidf ngram Vectors: ", accuracy)

# Bagging/Ensemble model- Random Forest
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
