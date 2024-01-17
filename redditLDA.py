#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 02:22:30 2024

@author: marwakasim
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from gensim import corpora
import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

file_path = ('reddit_wsb.csv')
reddit_data =  pd.read_csv(file_path)

#Convert 'created' column to datetime format
reddit_data['created_datetime'] = pd.to_datetime(reddit_data['created'], unit='s')

#Filter the data for the relevant time period (e.g., January 2021)
start_date = '2021-01-01'
end_date = '2021-01-31'
short_squeeze_data = reddit_data[(reddit_data['created_datetime'] >= start_date) & 
                                 (reddit_data['created_datetime'] <= end_date)]

#Preprocess text function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    return text

#Apply preprocessing to the title column
processed_titles = short_squeeze_data['title'].map(preprocess_text)

#Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_titles)

#Convert document into the bag-of-words (BoW) format
corpus = [dictionary.doc2bow(doc) for doc in processed_titles]

#Train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

#Display the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
    
lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)

#Display the visualization in a Jupyter Notebook
pyLDAvis.display(lda_display)

pyLDAvis.save_html(lda_display, 'lda_visualization.html')











