# ref https://www.kaggle.com/hemanth346/bbc-classification
'''
Dataset Overview

Consists of 2555 documents from the BBC news website corresponding to stories in five topic
areas from 2004-2005

Natural Classes: 5
business
entertainment
poltitcs
sport
tech

There seems to be 2 separate folders with same data, Ignored the second folder.
Dcouments corresponding to classes are inside respective folder name
Each class has around 400-500 documents in it

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import random
import nltk
from nltk.corpus import stopwords
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import linear_model, naive_bayes, svm

import joblib
data_dir = '../bbc_news_Data/bbc-fulltext (document classification)/bbc/'
CATEGORY = 'category'
DOCUMENT_ID = 'document_id'
TEXT = 'text'
TITLE = 'title'
STORY = 'story'


def clean_text(text):
    # decontraction: https://stackoverflow.com/a/47091490/7445772
    # specific
    text =re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", 'can not', text)

    #general
    text = re.sub(r"n\'t", 'not', text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'s", "is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # remove line breaks  \r \n \t remove from string
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    # remove stop words
    text = ' '.join(word for word in text.split() if word not in stopwords)

    # remove special words
    text = re.sub('[^A-Za-z0-9]+', " ", text)
    text = text.lower()
    return text

def show_confusion_matrix(prediction, y_test):
    # https://stackoverflow.com/a/48018785/7445772
    labels = ['tech','sport', 'business','entertainment','politics']
    cm = confusion_matrix(y_test, prediction, idx) #?? idx ?
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)  #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(labels, rotation=90);
    ax.yaxis.set_ticklabels(labels[::-1], rotation=90);
    plt.title('Confusion matrix of the classifier')
    plt.show()

# Reading data into Dataframes for easy overview of data and subsequent processing

frame = defaultdict(list)
for dir_name, _ , file_names in os.walk(data_dir):

    try:
        #file_names.remove('README.TXT')
        file_names.remove('.DS_Store')
    except:
        pass

    for file_name in file_names:
        frame[CATEGORY].append(os.path.basename(dir_name))
        name = os.path.splitext(file_name)[0]
        frame[DOCUMENT_ID].append(name)
        path = os.path.join(dir_name, file_name)
        # throw UnicodeDecodeError without encoding
        # Googled "UnicodeDecodeError: 'utf-8' codec can't decode by 0xa3 "
        # https://stackoverflow.com/questions/17219625/unicodedecodeerror-ascii-codec-cant-decode-byte-0xa3
        # https://www.google.com/search?ei=0xtkX8zLK9KUmAWcrIioBA&q=0xa3+utf+8&oq=0xa3&gs_lcp=CgZwc3ktYWIQAxgBMgQIABBDMgIIADICCAAyAggAMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeUABYAGCB9wZoAHAAeACAAY0BiAGNAZIBAzAuMZgBAKoBB2d3cy13aXrAAQE&sclient=psy-ab
        with open(path, 'r', encoding='unicode_escape') as file:
            frame['text'].append(file.read())

df = pd.DataFrame.from_dict(frame)

# exploring data
print(df.head(3))
print(df[CATEGORY].value_counts())

df.drop(0,axis=0, inplace=True)
print('document num :', len(df[DOCUMENT_ID].unique()))

# seems like incremental numbers for document names
sorted_ids = sorted(df[DOCUMENT_ID].unique())
print(sorted_ids[:10],sorted_ids[ -10:])

#show 5 row data

num = 5
sample = random.sample(range(df.text.shape[0]), num)
for idx in sample:
    print('*' *30)
    values = df.iloc[idx]
    print(DOCUMENT_ID + ':', values[DOCUMENT_ID])
    print(CATEGORY + ':', values[CATEGORY])
    print(TEXT + ': \n ' + '-'*7)
    print(values[TEXT])
    print('='*36)

'''
On brief observation, we can see that first line of text seems to be title and the 
next part is story/news article

- we can split the text column into 2 separate features title and story
- we can see there are financial symbols, punctuation marks, many stop words,
new lines and some words like doesn't didn't indside the text.
These has to be taken care during pre-processing
'''
# https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
text = df['text'].str.split('\n', n=1, expand=True)
df[TITLE] = text[0]
df[STORY] = text[1]
print(df.head(2))

# Univariate Analysis - Category
ax = sns.countplot(df.category)
title_obj = plt.title('Number of documents in each  category')
#
# ax.xaxis.label.set_color("green")
# ax.tick_params(axis='x', colors='green')
# ax.yaxis.label.set_color("green")
# ax.tick_params(axis='y', colors='green')
# plt.getp(title_obj)                    #print out the properties of title
# plt.getp(title_obj, 'text')            #print out the 'text' property for title
# plt.setp(title_obj, color='g')         #set the color of title to red
# plt.savefig('category.png')
plt.show()

# Univariate Analysis Title
# Googled "How to calculate number of words in a string in DataFrame" - https://stackoverflow.com/a/37483537/4084039

word_dict =dict(df[TITLE].str.split().apply(len).value_counts())
idx = np.arange(len(word_dict))
plt.figure(figsize=(20,10))
p1 = plt.bar(idx, list(word_dict.values()))
plt.ylabel('Number of documents')
plt.xlabel('Number of words in document title')
plt.title('Words for each title of the document')
plt.xticks(idx, list(word_dict.keys()))
plt.show()

# Lot of project titles have 4-6 word count, while there are very insignficant
cat_titles_word_count = defaultdict(list)
for category in df.category.unique():
    val = df[df[CATEGORY]==category][TITLE].str.split().apply(len).values
    cat_titles_word_count[CATEGORY] = val


# distribution of titles across categories
plt.boxplot(cat_titles_word_count.values())
keys = cat_titles_word_count.keys()
plt.xticks([i+1 for i in range(len(keys))], keys)
plt.ylabel('Word in document title')
plt.grid()
plt.show()

'''
 - Mean of title word count is 5 for all categories
 - Sports category titles seems to have smaller titles
 - All categories except for sports have 25%le to 75%le between 4 to 7 ----??? 
'''

# distribution of words in title
plt.figure(figsize=(12,4))
for key, value in cat_titles_word_count.items():
    sns.kdeplot(value, label=key, bw=0.6)
plt.legend()
plt.title('Distribution of words in title')
plt.show()

#=================

# Univariate Analysis - Story

category_story_word_count = defaultdict(list)
for category in df.category.unique():
    val = df[df[CATEGORY]==category][STORY].str.split().apply(len).values
    category_story_word_count[category] = val

# distribution of stories across categories
plt.boxplot(category_story_word_count.values())
plt.title('Distribution of words in stories across categories')
keys = category_story_word_count.keys()
plt.xticks([i+1 for i in range(len(keys))], keys)
plt.ylabel('Words in stories')
plt.grid()
plt.show()

# distribution of words in story
fig, axes = plt.subplots(2, 3, figsize=(15,8), sharey=True)
ax = axes.flatten()
plt.suptitle('Distribution of words in story')
for idx, (key, value) in enumerate(category_story_word_count.items()):
    sns.kdeplot(value, label=key, bw=0.6, ax=ax[idx])
plt.legend()
plt.show()

# Pre-processing

## Title pre_processing
samples = random.sample(range(len(df.title)), 10)
for idx in sample:
    print(df.title[idx])
    print('-' * 36)

## stop words
stopwords = stopwords.words('english')
print(stopwords)


processed_titles = []
for title in tqdm(df[TITLE].values):
    proecessed_title = clean_text(title)
    processed_titles.append(proecessed_title)

## tiles after processing
for idx in samples:
    print(processed_titles[idx])
    print('-' * 36)

# Story - Text preprocessing
processed_stories = []
for story in tqdm(df[STORY].values):
    processed_story = clean_text(story)
    processed_stories.append(processed_story)


for i in range(5):
    print(df.category.values[i])
    print(processed_titles[i])
    print(processed_stories[i])
    print('-'*100)


# Prepare Data for models
## Vectorizing Text data

### Title Bag of words

#### min_df :忽略掉词频严格低于定阈值的词

#### consider only the words which  appeared in at least 5 document titles
vectorizer = CountVectorizer(min_df=5)
title_bow = vectorizer.fit_transform(processed_titles)
print('Shape after one hot encoding :', title_bow.shape)

### Title - TFIDF
vectorizer =TfidfVectorizer(min_df=5)
title_tfidf = vectorizer.fit_transform(processed_titles)
print('Shape after tfidf one hot encoding ', title_tfidf.shape)

### Title n-gram
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=750, min_df=5)
title_tfidf_ngram = vectorizer.fit_transform( processed_titles)
print('Shape after tfidf ngram one hot encoding: ', title_tfidf_ngram)


## Story/Document - Bag of words(Bow)

### condisder only the words which appeared in at least 10 document
vectorizer = CountVectorizer(min_df=10)
story_bow = vectorizer.fit_transform(processed_stories)
print('Story Shape after one hot encoding', story_bow.shape)

### Story / Document - TFIDF

vectorizer = TfidfVectorizer(min_df=10)
story_tfidf = vectorizer.fit_transform(processed_stories)
print('Story tfidf Shape after one hot encoding', story_tfidf.shape)

vectorizer = TfidfVectorizer(analyzer = "word", ngram_range =(1, 4),max_features = 7500, min_df=10)
story_tfidf_ngram = vectorizer.fit_transform(processed_stories)
print("Story tfidf_ngram Shape after one hot encodig ",story_tfidf_ngram.shape)

## Vectorizing Categorical data -- Target feature


vectorizer = LabelEncoder()
category_onehot = vectorizer.fit_transform(df['category'].values)
# print(vectorizer.get_feature_names())

print("Category Shape after one hot encoding : ",category_onehot.shape)

# Model input
## Merging the features
## preparing target

X_bow = hstack((title_bow, story_bow))
X_tfidf = hstack((title_tfidf, story_tfidf))
X_ngram = hstack((title_tfidf_ngram, story_tfidf_ngram))

print(X_bow.shape, X_tfidf.shape, X_ngram.shape)
print(type(category_onehot), type(X_bow))

# Data Mdoeling



# Use bow vectors
X = X_bow.toarray()
y = category_onehot
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=9)

model = linear_model.LogisticRegression(max_iter=400)
# model.fit(X_train, y_train)
# LR_prediction = model.predict(X_test)
#
# model = naive_bayes.MultinomialNB()
# model.fit(X_train,y_train)
# NB_prediction = model.predict(X_test)
#
# model = svm.SVC()
# model.fit(X_train,y_train)
# SVM_prediction = model.predict(X_test)



# print('Accuracy with BOW vectors \n'+'-'*15)
# print(f'Using Logistic regression:{accuracy_score(LR_prediction,y_test)}')
# print(f'Using Naive Bayes:{ accuracy_score(NB_prediction, y_test)}')
# print(f'Using Support Vector Machines : {accuracy_score(SVM_prediction, y_test)}')
# show_confusion_matrix(LR_prediction, y_test)
# show_confusion_matrix(NB_prediction, y_test)
# show_confusion_matrix(SVM_prediction, y_test)



# Using TFIDF vectors      ---copy paste , i not wrirte,  the confsuon_matrix
# have problem from source code
X = X_tfidf.toarray()
y = category_onehot
print(X.shape, y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=9)

model = linear_model.LogisticRegression(max_iter=500)
#model.fit(X_train,y_train)
#joblib.dump(model, 'LR_model')
#LR_prediction = model.predict(X_test)
#
# model = naive_bayes.MultinomialNB()
# model.fit(X_train,y_train)
# NB_prediction = model.predict(X_test)
#
# model = svm.SVC()
# model.fit(X_train,y_train)
# SVM_prediction = model.predict(X_test)

# print('Accuracy with TF IDF vectors \n'+'-'*15)
# print(f'Using Logistic regression : {accuracy_score(LR_prediction, y_test)}')
# print(f'Using Naive Bayes : {accuracy_score(NB_prediction, y_test)}')
# print(f'Using Support Vector Machines : {accuracy_score(SVM_prediction, y_test)}')
# show_confusion_matrix(LR_prediction, y_test)
# show_confusion_matrix(NB_prediction, y_test)
# show_confusion_matrix(SVM_prediction, y_test)

# tfidf ngram vector
vframe = defaultdict(list)

with open('./varify.txt', 'r', encoding='unicode_escape') as file:
    vframe['text'].append(file.read())
vdf = pd.DataFrame.from_dict(vframe)

vstories = []
for story in tqdm(vdf['text'].values):
    text = clean_text(story)
    vstories.append(text)
text = vdf['text'].str.split('\n', n=1, expand=True)
vdf[TITLE] = text[0]
vdf[STORY] = text[1]
processed_titles_ver = []
for title in tqdm(vdf[TITLE].values):
    processed_title = clean_text(title)
    processed_titles_ver.append(proecessed_title)
processed_stories_ver = []
for story in tqdm(vdf[STORY].values):
    processed_story = clean_text(story)
    processed_stories_ver.append(processed_story)

vectorizer =TfidfVectorizer()
title_tfidf = vectorizer.fit_transform(processed_titles_ver)

vectorizer = TfidfVectorizer()
story_tfidf = vectorizer.fit_transform(processed_stories_ver)
X_tfidf =  hstack((title_tfidf, story_tfidf))
X = X_tfidf.toarray()


model = linear_model.LogisticRegression(max_iter=500)
#model.fit(X_train ,y_train)
#joblib.dump(model, 'LR_model')
model = joblib.load('./model/LR_model')