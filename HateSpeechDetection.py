# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:44:24 2023

@author: khoac
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stemmer = nltk.SnowballStemmer('english')
from nltk.util import pr
import string
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
stopword = set(stopwords.words('english'))
df = pd.read_csv('train.csv')



df['labels'] = df['label'].map({0:"Neutral",1:"Hateful"})
df = df[['tweet','labels']]
df = df.dropna(subset=['labels'])


def cleanTwts(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9]+','', text)
    text = re.sub(r'#','', text)
    text = re.sub(r'RT[\s]+','', text)
    text = re.sub(r'https?:\/\/\S+','', text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = ' '.join(text)
    return text

#Clean tweets
df["tweet"] = df["tweet"].apply(cleanTwts)

#%%

x = np.array(df['tweet'])
y = np.array(df['labels'])

cv = CountVectorizer()
x = cv.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

#%%
df2 = pd.read_csv('2016_US_election_tweets.csv')
print(df2.info())

#%%
df2['text'] = df2['text'].apply(cleanTwts)
df2 = df2[df2.text != 'nan']
df2['text'] = df2['text'].apply(cleanTwts)
df2['labels'] = pd.DataFrame(clf.predict(cv.transform(df2['text'])))

#%%

hateful = df2[df2.labels == 'Hateful']
hateful = hateful[['text','labels']]
neutral = df2[df2.labels == 'Neutral']
neutral = neutral[['text','labels']]

#%%

#Calculates the number and percentage of offensive, hateful, and neutral tweets
def getPercentages(df):
    
    hatefulTweets = df[df.labels == 'Hateful']
    
    neutralTweets = df[df.labels == 'Neutral']
    
    percentHateful = round(hatefulTweets.shape[0]/df.shape[0]*100,2)
    print ('Hateful tweets: ' +str(hatefulTweets.shape[0]) + ', ' + str(percentHateful) + '% of total')
    
    percentNeutral = round(neutralTweets.shape[0]/df.shape[0]*100,2)
    print ('Neutral tweets: ' +str(neutralTweets.shape[0]) + ', ' + str(percentNeutral) + '% of total')

getPercentages(df2)


#%%
hateful = df2[df2.labels == 'Hateful']
hateful['created_at'] = pd.to_datetime(hateful['created_at'], errors= 'coerce')
hateful.dropna(subset = ['created_at'])
hateful['date'] = hateful['created_at'].dt.date

grouped = hateful.groupby(by='date')['labels'].count()

fig, ax = plt.subplots(figsize=(8, 6))
plt.title('Hateful Sentiment vs Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
ax.plot(grouped.index, grouped)