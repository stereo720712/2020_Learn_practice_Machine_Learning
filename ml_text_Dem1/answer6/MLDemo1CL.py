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

CATEGORY = 'category'
DOCUMENT_ID = 'document_id'
TITLE = 'title'
INPUT = 'input'
LABEL = 'label'

class MLDemo1CL(object):




    def __init__(self):
        self._stopword_e = stopwords.words("english")
        self.DATA_DIR = os.sep.join(['..','bbc_news_Data', 'bbc-fulltext (document classification)', 'bbc'])
        self.df = None
        self.vdf = None

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

    def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100,
                     figure_size=(24.0, 16.0), title=None, title_size=40, image_color=False):

          stopwords = set(STOPWORDS)  # ?
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
              plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})

          else:
              plt.imshow(wordcloud)
              plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})

          plt.axis('off')
          plt.tight_layout()

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
                frame[LABEL].append(os.path.basename(dir_name))
                name = os.path.splitext(file_name)[0]
                frame[DOCUMENT_ID].append(name)
                path = os.path.join(dir_name, file_name)
                with open(path, 'r',encoding='unicode_escape') as file:
                    frame[INPUT].append(file.read())
        self.df = pd.DataFrame.from_dict(frame)


    def showSourceData(self):
        ''' show the source data frame'''
        print(self.df.head(6))

    def showSourceDataCountGraph(self):
        '''show the source data count at every category'''
        pass
    def showWordFashionGraph(self):
        '''word_~~ cloud'''
        pass
    def data_process(self):
        '''label , input , show processing'''
        pass

    def model_train(self):
        '''traiing model'''
        pass

    def model_test(self):
          ''' show test predict result'''

    def show_confusion_matrix(prediction, y_test):
          '''from bbc classificaiton, wait for find data to show'''
          pass
