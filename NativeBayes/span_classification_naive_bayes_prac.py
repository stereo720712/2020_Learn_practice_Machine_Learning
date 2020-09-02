import matplotlib.pyplot as plt
import pandas as pd
import string
import codecs
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  naive_bayes as bayes
from sklearn.model_selection import train_test_split

#open file
file_path = "./"
email_frame = pd.read_excel(os.path.join(file_path, "chinesespam.xlsx"),0)

print("data_shape:", email_frame.shape)
print("spams_in_rows: ", email_frame.loc[email_frame["type"] == "spam"].shape[0])
print("ham in rows: ", email_frame.loc[email_frame['type'] == 'ham'].shape[0])

# load stop words , for what determine
stop_words = codecs.open(os.path.join(file_path,'stopwords.txt'),"r",'UTF-8').read().split('\r\n')

# cut words and process that
processed_texts = []

# 去掉 stopword 中的字彙 ？！～
for text in email_frame['text']: # one context
  words = []
  seq_list = jieba.cut(text)
  for seq in seq_list:
      if(seq.isalpha()) & (seq not in stop_words):
          words.append(seq)
  sentence = ' '.join(words)
  processed_texts.append(sentence)
email_frame["text"] = processed_texts

print(email_frame.head(3))

# transform text to sparse matrix
def transform_text_to_sparse_matrix(texts):
     #  https://ppt.cc/fko20x
    vectorizer = CountVectorizer(binary=False)
    vectorizer.fit(texts)
    # inspect vocabulary
    vocabulary = vectorizer.vocabulary_
    print('There are ', len(vocabulary), "word features")
    vector = vectorizer.transform(texts)
    result = pd.DataFrame(vector.toarray()) # embedding vector matrix
    keys = []
    values = []

    for key, value in vectorizer.vocabulary_.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame(data={"key":keys, "value":values})
    col_names = df.sort_values("value")["key"].values
    result.columns = col_names

    return result

text_matrix = transform_text_to_sparse_matrix(email_frame["text"])
print(text_matrix.head(3))

# pop freq words https://ppt.cc/fBGWKx
features = pd.DataFrame(text_matrix.apply(sum,axis=0))
print(features.head(5))
extracted_features = [features.index[i] for i in range(
    features.shape[0]) if features.iloc[i, 0] > 5]

text_matrix = text_matrix[extracted_features]
print('there are ', text_matrix.shape[1], "word features")


# train_data # test data
train, test, train_label, test_label = train_test_split(text_matrix, email_frame["type"], test_size=0.2)

# train model
clf = bayes.BernoulliNB(alpha=1,binarize=True) #?
model = clf.fit(train, train_label)

print(model.score(test,test_label))

res = model.predict(test)

print(res)

