# ref https://www.kaggle.com/rissuuuu/notebook7d2b6a9283


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import  stopwords
import  seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import random
from collections import defaultdict

data_rootPath = '../bbc_news_Data'
dataPath = '../bbc_news_Data/bbc-fulltext (document classification)/bbc/'

#-----------------------------------------------------
# for dirname, _, filenames in os.walk(data_rootPath):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#

#print(os.listdir(dataPath))

#--------------------------------------------------------

dicts = defaultdict(list)

for dir_name, _, file_names in os.walk(dataPath):

    try:
        file_names.remove('README.TXT')
        file_names.remove('.DS_Store')
    except:
        pass

    for file in file_names:
        dicts['categories'].append(os.path.basename(dir_name))
        name = os.path.splitext(file)[0]
        dicts['doc_id'].append(name)
        path = os.path.join(dir_name, file)


        with open(path, 'r', encoding='latin-1') as file:
            dicts['text'].append(file.read())

df = pd.DataFrame.from_dict(dicts)
print(df.head(5))

# remove README.txt
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop('doc_id', axis=1, inplace=True)

print(df['text'][1])

# random sample 5 number
random.sample(range(df.text.shape[0]), 5)
ax = sns.countplot(df['categories'])
for i, j in enumerate(df['categories'].value_counts().values):
    ax.text(i-0.3, 100, j, fontsize=20)
plt.title('Value counts for each categories')
plt.show()

# split by '/n '
test = ' '.join(df['text'][0].split('/n')[1:])
' '.join(df['text'][0].split('\n')[1:])
' '.join(df['text'][0].split('\n'))
print(test)

def returnTitle(data):
    data = data.split('\n')[0]
    return data
def returnArticle(data):
    data = ' '.join(data.split('\n')[1:])
    return data

print(df['text'][0].split('\n')[0])
print('---------------------------')
print(df['text'][0].split('\n')[1:])

print('---------------------------')

df['title'] = df['text'].apply(lambda x:returnTitle(x))
df['article'] = df['text'].apply(lambda x:returnArticle(x))

print(df.head())

wordlength = dict(df['title'].str.split().apply(len).value_counts())
plt.bar(list(wordlength.keys()), list(wordlength.values()))
plt.xlabel('length of word title')
plt.ylabel('Documents  Count')
plt.title('Number of words in each title')
plt.show()

import nltk

words = {}

for i in df['title']:
    for j in i.split():
        words[j] = 0
for i in df['title']:
    for j in i.split():
        words[j]+=1

toptitles = nltk.FreqDist(words)
words = list(toptitles.keys())[:15]
vals = list(toptitles.values())[:15]
plt.figure(figsize=(8,10))
ax = sns.barplot(vals, words)
for i, j in enumerate(vals):
    ax.text(8, i+.2, j, fontsize=15)

plt.title("Maximum occurance of word in title")
plt.show()

categories = defaultdict(list)
for i in df['categories'].unique():
    temp = df[ df['categories'] == i ]['article'].str.split().apply(len).values
    categories[i]= temp
# ========================================????
plt.boxplot(categories.values())
plt.xticks(range(5), categories.keys())
plt.show()


for i in categories.keys():
    plt.plot(categories[i])
    plt.title(i)
    plt.show()


# i think no ok
def Clean(text):
    text = text.split()
    text = [i.lower() for i in text if i.lower() not in stopwords.words('english')]
    text = ' '.join(text)
    text = re.sub('[^A-Za-z0-9]]+'',', ' ', text)
    text = text.lower()
    return text

print(df.head(1))
df['article'] = df['article'].apply(lambda x:Clean(x))
print(df.head(1))



#label encoding
from sklearn.preprocessing import LabelEncoder
l_enc = LabelEncoder()

df['categories'] = l_enc.fit_transform(df['categories'])
#df.head()

# word vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['article'])
y = df['categories']

from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier #!!!!!!!!!!!!
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier

models = [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(),
          MultinomialNB(), XGBClassifier(), SVC(), PassiveAggressiveClassifier()]
scores = pd.DataFrame({'Model':[], 'Train_Score':[], 'Test_Score':[]})
for j, i in enumerate(models):
    i.fit(X_train, y_train)
    #i.save_model()
    scores.loc[j,:] = [i, i.score(X_train, y_train), i.score(X_test, y_test)]

print(scores)




