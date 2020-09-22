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
df.drop(0,axis=0, inplace=True) # readme.txt
text = df[TEXT].str.split('\n', n=1, expand=True)
df[TITLE] = text[0]
df[STORY] = text[1]
df.drop(TEXT,axis=1,inplace=True)
df.head(3)

# ================================verify data
vframe = defaultdict(list)
with open('./varify.txt', 'r', encoding='unicode_escape') as file:
    vframe['text'].append(file.read())
vdf = pd.DataFrame.from_dict(vframe)
vtext = vdf[TEXT].str.split('\n', n=1, expand=True)
vdf[TITLE] = text[0]
vdf[STORY] = text[1]


#label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[CATEGORY])

# X converting
tfidf_conveter = TfidfVectorizer(analyzer='word', max_features=10000)



## stop words
stopwords = stopwords.words('english')
print(stopwords)