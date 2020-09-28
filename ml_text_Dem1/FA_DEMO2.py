'''
From FA_DEMO2.ipynb

'''

# LIB
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
DATA_DIR = os.sep.join(['bbc_news_Data','bbc-fulltext (document classification)','bbc'])
CATEGORY = 'category'
DOCUMENT_ID = 'document_id'
# TEXT = 'text'
TITLE = 'title' # remove this element for the demo
# STORY = 'story'
INPUT = 'input'
LABEL = 'label'

# READ Data from text
frame = defaultdict(list)
for dir_name, _, file_names in os.walk(DATA_DIR):
    try:
        file_names.remove('README.TXT')
        file_names.remove('.DS_Store')
    except:
        pass
    for file_name in file_names:
        frame[LABEL].append(os.path.basename(dir_name))
        name = os.path.splitext(file_name)[0]
        frame[DOCUMENT_ID].append(name)
        path = os.path.join(dir_name, file_name)
        with open(path,'r', encoding='unicode_escape') as file:
            frame[INPUT].append(file.read())
df = pd.DataFrame.from_dict(frame)

# DATA Encoding

# LABEL
df[LABEL].unique()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL].values)
y


# TEXT
nltk.download('stopwords')
stopword_e = stopwords.words('english')
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
    text = ' '.join(word for word in text.split() if word not in stopword_e)

    # remove special words
    text = re.sub('[^A-Za-z0-9]+', " ", text)
    text = text.lower()
    return text

input_processed = []
for txt in tqdm(df[INPUT].values):
    txt_p = clean_text(txt)
    input_processed.append(txt_p)

#Vect.
text_transformer = TfidfVectorizer(min_df=2)
X = text_transformer.fit_transform(input_processed)
X = X.toarray()
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)

# Model trtain
model = linear_model.LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
joblib.dump(model,'LR_model')
#model = joblib.load('LR_model')
#lr_prediction = model.predict(X_test)
#print('accurancy: ',accuracy_score(lr_prediction,y_test))

#Verify txt test
vframe = defaultdict(list)
with open('varify.txt','r', encoding='unicode_escape')as file:
    vframe['txt'].append(file.read())
vdf = pd.DataFrame.from_dict(vframe)
v_input = []
for txt in vdf['txt']:
    txt_p = clean_text(txt)
    for i in range(2225):
        v_input.append(txt_p)
X_v = text_transformer.fit_transform(v_input)
X_v = X_v.toarray()

print(X_v)

model_l = joblib.load('LR_model')
res = model_l.predict(X_v)
print(res)
