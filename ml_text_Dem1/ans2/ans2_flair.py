import numpy as np
import pandas as pd
import os

import  pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import seaborn as sns
import eli5
from  IPython.display import Image
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
        #filenames.remove('README.TXT')
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
                    title.append(line.replace('\n',''))
                    first_line = False
                else:
                    in_text = in_text + ' ' + line.replace('\n', '')
            text.append(in_text)

## in demo you can delete the title data

dataZip = zip(directory, file, title, text, label)
dataList = list(dataZip)

fullDf = pd.DataFrame(dataList,columns=['directory', 'file', 'title', 'text', 'label'])
df = fullDf.filter(['title', 'text', 'label'], axis=1)
df['text'] = df['title'] + df['text']
df = df.drop(columns=['title'])
df = df[['label', 'text']]
print('FullDf: ', fullDf.shape)
print('DF: ', df.shape)


# data split for train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)
dev, test = train_test_split(test,test_size=0.5)
print('Train DF: ', train.shape)
print('Dev DF: ', dev.shape)
print('Test DF: ', test.shape )

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
#download corpus

from torch.optim.adam import Adam
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import  TransformerDocumentEmbeddings
from flair.models import  TextClassifier
from flair.trainers import ModelTrainer

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)
trainer.train('classify/model',
              learning_rate=3e-5,
              mini_batch_size=16,
              mini_batch_chunk_size=4, # # optionally set this if transformer is too much for your machine
              max_epochs=5
              )

from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('classify/model/loss.tsv')

from flair.nn import  Model
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
