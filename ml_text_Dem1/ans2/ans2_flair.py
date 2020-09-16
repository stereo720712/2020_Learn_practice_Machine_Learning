import numpy as np
import pandas as pd
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import seaborn as sns
import eli5
from IPython.display import Image

from torch.optim.adam import Adam
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from matplotlib import pyplot as plt


for dirname, _, filenames in os.walk(
        '../bbc-full-text-document-classification/bbc-fulltext (document classification)/bbc/'):
    # for filename in filenames:
    # print(os.path.join(dirname,filename))
    pass

'''
# data preprossing
data: title text label
1.Walk through each folder 
2.Pick the complete path , directory and filename
3.Read a file into a string line by line
4.Pick the first line as title
5.Set the directory name as label
'''

# step 1 get  file details
directory = []
file = []
title = []
text = []
label = []
datapath = '../bbc_news_Data/bbc-fulltext (document classification)/bbc/'
for dirname, _, filenames in os.walk(datapath):
    print('Directory:', dirname)
    print('Subdir:', dirname.split('/'[-1]))
    # remove the Readme.txt file
    # will not find file in the second iteration so we skip the error
    try:
        filenames.remove('README.TXT')
        filenames.remove('.DS_Store')
    except:
        pass

    for filename in filenames:
        directory.append(dirname)
        file.append(filename)
        label.append(dirname.split('/')[-1])
        print(filename)
        full_path_file = os.path.join(dirname, filename)
        with open(full_path_file, 'r', encoding='utf-8', errors='ignore') as infile:
            in_text = ''
            first_line = True
            for line in infile:
                if first_line:
                    title.append(line.replace('\n', ''))
                    first_line = False
                else:
                    in_text = in_text + ' ' + line.replace('\n', '')
            text.append(in_text)

## in demo you can delete the title data


fullDf = pd.DataFrame(list(zip(directory, file, title, text, label)),
                      columns=['directory', 'file', 'title', 'text', 'label'])

df = fullDf.filter(['title', 'text', 'label'], axis=1)
df['text'] = df['title'] + df['text']
df = df.drop(columns=['title'])
df = df[['label', 'text']]
print('FullDf: ', fullDf.shape)
print('DF: ', df.shape)

# data split for train and test
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3)
dev, test = train_test_split(test, test_size=0.5)
print('Train DF: ', train.shape)
print('Dev DF: ', dev.shape)
print('Test DF: ', test.shape)

import os

os.makedirs('classify', exist_ok=True)
os.makedirs('classify/model', exist_ok=True)
os.makedirs('classify/data', exist_ok=True)
train.to_csv('classify/data/train.csv')
dev.to_csv('classify/data/dev.csv')
test.to_csv('classify/data/test.csv')

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# this is the folder in which train , test  and dev files reside
data_folder = 'classify/data'

# column format indicating which columns hold the next labels
column_name_map = {2: 'text', 1: 'label'}

# load corpus containing training , test and dev data and if CSV has a header
#  you can skip it

corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter=',')
label_dict = corpus.make_label_dictionary()

from flair.embeddings import TransformerDocumentEmbeddings

document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)
# download corpus


classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)
trainer.train('classify/model',
              learning_rate=3e-5,
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # # optionally set this if transformer is too much for your machine
              max_epochs=2 # original is 5
              )

from flair.visual.training_curves import Plotter

plotter = Plotter()
plotter.plot_training_curves('classify/model/loss.tsv')

from flair.nn import Model

model = TextClassifier.load('classify/model/final-model.pt')
result = model.predict(corpus.test)

y_test = []
y_pred = []
for r in result:
    y_test.append(r.labels[0].value)
    y_pred.append(r.labels[1].value)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = label_dict.get_items()
cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=labels, normalize='true')
cm_display = ConfusionMatrixDisplay(cm, display_labels=labels).plot()


# eda
'''
1.find out the distribution of labels in our data
2. The distribution of sentence lengths
3. The frequency of words

'''

fullDf = pd.DataFrame(list(zip(directory, file, title, text, label)),
                      columns=['directory', 'file', 'text', 'label'])
df = fullDf.filter(['title', 'text', 'label'], axis=1)
train, text = train_test_split(df, test_size=0.2)
print('Train DF:', train.shape)
print('Test DF:', test.shape)
sns.countplot(df['label'])
plt.title('Data: Target distribution')

plt.subplot(1, 2)
plt.subplot(1, 2, 1)
df['text'].apply(lambda x: len(x.split())).plot(kind='hist')
plt.yscale('log') #?
plt.title('Text length in words')
plt.subplot(1, 2, 2)
df['title'].apply(lambda x:len(x.split())).plot(kind='hist')
plt.yscale('log')
plt.title('title length in words')

from wordcloud import  WordCloud, STOPWORDS, ImageColorGenerator

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100,
                   figure_size=(24.0,16.0), title=None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS) #?
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='blilnear')
        plt.title(title, fontdict={'size': title_size, 'verticalalignment':'bottom'})

    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size':title_size, 'color':'black', 'verticalalignment':'bottom'})

    plt.axis('off')
    plt.tight_layout()

plot_wordcloud(df['title'], title='Word Cloud of Titles')

# train model ????!!?!?! 上面不是跑過拉 幹你娘咧
text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True, max_features=150000)

X_train_text = text_transformer.fit_transform(train['text'])
X_test_text = text_transformer.transform(test['text'])

X_train = X_test_text
X_test = X_test_text

print("X Train DF: ",X_train.shape)
print("X Test DF: ", X_test.shape)

# The model  ,
# At this stage let us take the simpleest model a Logistic Regression model, i will use sclkit learns implementation
from  sklearn.model_selection import StratifiedKFold
logit = LogisticRegression(C=5e1,solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

#%%time
cv_results = cross_val_score(logit, X_train, train['label'], cv=skf,
                            scoring='f1_marco')
print(cv_results,cv_results.mean())
logit.fit(X_train, train['label'])

# ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions.
eli5.show_weights(estimator=logit, feature_names=text_transformer.get_feature_names(),top=(50,5))


