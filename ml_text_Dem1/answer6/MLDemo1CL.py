'''
For Maching Learning and FA Demo Class

'''

import os
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
CATEGORY = 'category'
DOCUMENT_ID = 'document_id'
TITLE = 'title'
STORY = 'story'
LABEL = 'label'

class MLDemo1CL(object):




    def __init__(self):
        self._stopword_e = stopwords.words("english")
        self.DATA_DIR = os.sep.join(['..','bbc_news_Data', 'bbc-fulltext (document classification)', 'bbc'])
        self.df = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(min_df=3)
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_test_predict = None


    def clean_text(self, text):
          # decontraction: https://stackoverflow.com/a/47091490/7445772
          # specific
          text = re.sub(r"won\'t", "will not", text)
          text = re.sub(r"can\'t", 'can not', text)

          # general
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
          text = ' '.join(word for word in text.split() if word not in self._stopword_e)

          # remove special words
          text = re.sub('[^A-Za-z0-9]+', " ", text)
          text = text.lower()
          return text

    def plot_wordcloud(text, mask=None, max_words=20000, max_font_size=100,
                     figure_size=(30.0, 24.0), title=None, title_size=40, image_color=False):

          stopwords = set(STOPWORDS)  # ?
          more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
          stopwords = stopwords.union(more_stopwords)

          wordcloud = WordCloud(background_color='black',
                                stopwords=stopwords,
                                max_words=max_words,
                                max_font_size=max_font_size,
                                random_state=42,
                                width=1024,
                                height=768,
                                mask=mask)
          wordcloud.generate(str(text))

          plt.figure(figsize=figure_size)
          if image_color:
              image_colors = ImageColorGenerator(mask)
              plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='blilnear')
              plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})

          else:
              plt.imshow(wordcloud)
              plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})

          plt.axis('off')
          plt.tight_layout()

    def show_confusion_matrix(self, prediction, y_test):
        # https://stackoverflow.com/a/48018785/7445772
        labels = ['tech', 'sport', 'business', 'entertainment', 'politics']
       # cm = confusion_matrix(y_test, prediction, idx)  # ?? idx ?
        cm = confusion_matrix(y_test,prediction)
        print(cm)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix');
        ax.xaxis.set_ticklabels(labels, rotation=90);
        ax.yaxis.set_ticklabels(labels[::-1],rotation=0);
        #ax.yaxis.set_rotation(0)

        plt.title('Confusion matrix of the classifier')
        plt.show()

    def loadData(self):
        '''load data from text file and return dataframe '''
        frame = defaultdict(list)
        for dir_name, _, file_names in os.walk(self.DATA_DIR):
            try:
                file_names.remove('README.TXT')
                file_names.remove('.DS_Store')
            except:
                pass
            for file_name in file_names:
                frame[CATEGORY].append(os.path.basename(dir_name))
                name = os.path.splitext(file_name)[0]
                frame[DOCUMENT_ID].append(name)
                path = os.path.join(dir_name, file_name)
                with open(path, 'r',encoding='unicode_escape') as file:
                    frame[STORY].append(file.read())
        self.df = pd.DataFrame.from_dict(frame)
        print('Loading data finish')

    def showSourceData_nb(self):
        ''' show the source data frame in notebook'''
        self.df.head()

    def showSourceDataCountGraph(self):
        '''show the source data count at every category'''
        if self.df is None:
            print('Loading data...')
            self.loadData()
        ax = sns.countplot(self.df[CATEGORY])
        title_obj = plt.title('Number of documents in each  category')
        plt.getp(title_obj)  # print out the properties of title
        plt.getp(title_obj, 'text')  # print out the 'text' property for title
        plt.setp(title_obj, color='g')  # set the color of title to red
        plt.savefig('category.png')
        plt.show()



    def showWordFashionGraph(self):
        '''word_~~ cloud'''
        if self.df is None:
            print('Loading data...')
            self.loadData()
        x = self.df.story[0]
        self.plot_wordcloud(self.df[self.df[CATEGORY] == 'sport'][STORY], title='STORY  WORD SHOW')
    def data_process(self):
        '''label , input , show processing'''
        self.y = self.label_encoder.fit_transform(self.df[CATEGORY].values)
        # save
        pickle.dump(self.label_encoder, open('label_encoder','wb'))

        self.df[STORY] = self.df[STORY].apply(lambda x: self.clean_text(x))
        self.X = self.vectorizer.fit_transform(self.df[STORY])
        pickle.dump(self.vectorizer,open('vectorizer','wb'))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,test_size=0.2)
        print(self.X[0])

    def model_train(self):
        '''traiing model'''
        model = linear_model.LogisticRegression(max_iter=20000)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model,open('lr_model','wb'))

    def model_test(self):
        ''' show test predict result'''
        model = pickle.load(open('lr_model', 'rb'))
        self.y_test_predict = model.predict(self.X_test)
        print(self.label_encoder.inverse_transform(self.y_test_predict))

    def show_confusion_matrix_demo(self):
        '''from bbc classificaiton, wait for find data to show '''
        self.show_confusion_matrix(self.y_test_predict, self.y_test)

    def predict(self, file_path):
      '''load txt file and show result'''
      vframe = defaultdict(list)
      with open(file_path,'r',encoding='unicode_escape') as file:
          vframe['txt'].append(file.read())
      vdf = pd.DataFrame.from_dict(vframe)
      vdf['txt'] = vdf['txt'].apply(lambda x: self.clean_text(x))
      X_v = self.vectorizer.transform(vdf['txt'])
      model = pickle.load(open('lr_model','rb'))
      res = model.predict(X_v)
      print('預測結果是： ' , self.label_encoder.inverse_transform(res))

