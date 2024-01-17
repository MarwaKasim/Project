#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 02:02:48 2024

@author: marwakasim
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from textblob import TextBlob
import numpy as np


file_path = ('reddit_wsb.csv')
reddit_data =  pd.read_csv(file_path)

#Convert the 'created' column to datetime
reddit_data['created_datetime'] = pd.to_datetime(reddit_data['created'], unit='s')

# Define a function to calculate sentiment polarity
def calculate_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return np.nan

#Apply sentiment analysis on the title and body columns
reddit_data['title_sentiment'] = reddit_data['title'].apply(calculate_sentiment)
reddit_data['body_sentiment'] = reddit_data['body'].apply(calculate_sentiment)


#Display the updated dataset
reddit_data.head()

#Define the time period for the 2021 short-squeeze
start_date = '2021-01-01'
end_date = '2021-01-31'

#Re-run the code to convert the 'created' column to datetime
reddit_data['created_datetime'] = pd.to_datetime(reddit_data['created'], unit='s')


#Filter the dataset for the relevant time period
short_squeeze_data = reddit_data[(reddit_data['created_datetime'] >= start_date) & 
                                 (reddit_data['created_datetime'] <= end_date)]

#Topic Analysis: Generate a Word Cloud for the titles and bodies
combined_text = ' '.join(short_squeeze_data['title'].dropna() + ' ' + short_squeeze_data['body'].fillna(''))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(combined_text)

#Sentiment Analysis Summary: Distribution of Sentiment Scores
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(short_squeeze_data['title_sentiment'].dropna(), bins=30, kde=True, ax=ax[0])
ax[0].set_title('Distribution of Title Sentiment Scores')
sns.histplot(short_squeeze_data['body_sentiment'].dropna(), bins=30, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Body Sentiment Scores')

plt.show()

#Display the Word Cloud
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


