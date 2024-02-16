'''
This is a sentiment analysis program about tweets with 
the #CovidVaccine created from August 1st,2020 to March 4th, 2021.
'''
    
import tweepy
import pandas as pd
import numpy as np
import textblob
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
stemmer = nltk.SnowballStemmer('english')
import re
import os
plt.style.use('fivethirtyeight')
stopword = set(stopwords.words('english'))

tweets = pd.read_csv('2020hashtag_joebiden.csv')

#This function gets rid of @mentions, RT/ tags, and URLs
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
tweets["text"] = tweets["text"].apply(cleanTwts)



#%%
#This function creates a wordcloud of the most frequent words in 4 groups of tweets
def wordCloud(df):
    stopwords = set(STOPWORDS)
    stopwords.add("t")
    stopwords.add("co")
    stopwords.add("https")
    stopwords.add("will")
    stopwords.add("people")
    stopwords.add("amp")
    stopwords.add("time")
    stopwords.add("got")
    stopwords.add("now")
    stopwords.add("got")
    stopwords.add("say")
    stopwords.add("getting")
    stopwords.add("day")
    stopwords.add("today")
    stopwords.add("COVID")
    stopwords.add("vaccine")
    stopwords.add("COVID19")
    stopwords.add("CovidVaccine") 
        
    #Change to .csv format
    text = df['text'].to_csv()
    tweets_wc = WordCloud(
        background_color = 'white',
        max_words = 1000,
        stopwords = stopwords
        )
    tweets_wc.generate(text)
    plt.figure(figsize=(18,18))
    plt.imshow(tweets_wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
#%%

#This functions gets the subjectivity of each tweets
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#This function gets the polarity of each tweets
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


tweets['subjectivity'] = tweets['text'].apply(getSubjectivity)
tweets['polarity'] = tweets["text"].apply(getPolarity)


#This function provides an analysis based on polarity score
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
tweets['analysis'] = tweets['polarity'].apply(getAnalysis)
#%%

#Prints all the postive tweets


#%%

#Plots Polarity vs Subjectivity
def plotAnalysis(df):
    plt.figure(figsize=(6,4))
    for i in range (0, df['polarity'].shape[0]):
        plt.scatter(df['polarity'][i], df['subjectivity'][i],color = 'Blue')
    
    plt.title ('2020_hashtag_donaldtrump Polarity vs Subjectivity')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()

#%%
#Calculates the number and percentage of positive, negative, and neutral tweets
def getPercentages(df):
    posTweets = df[df.analysis == 'Positive']
    posTweets = posTweets['text']
    
    negTweets = df[df.analysis == 'Negative']
    negTweets = negTweets['text']
    
    neuTweets = df[df.analysis == 'Neutral']
    neuTweets = neuTweets['text']
    
    percentPositive = round(posTweets.shape[0]/df.shape[0]*100,2)
    print ('Positive tweets: ' +str(posTweets.shape[0]) + ', ' + str(percentPositive) + '% of total')
    
    percentNegative = round(negTweets.shape[0]/df.shape[0]*100,2)
    print ('Negative tweets: ' +str(negTweets.shape[0]) + ', ' + str(percentNegative) + '% of total')
    
    percentNeutral = round(neuTweets.shape[0]/df.shape[0]*100,2)
    print ('Neutral tweets: ' +str(neuTweets.shape[0]) + ', ' + str(percentNeutral) + '% of total')

getPercentages(tweets)
#%%

negative = tweets[tweets.analysis == 'Positive']
negative['created_at'] = pd.to_datetime(negative['created_at'], errors= 'coerce')
negative.dropna(subset = ['created_at'])
negative['date'] = negative['created_at'].dt.date

grouped = negative.groupby(by='date')['labels'].count()

fig, ax = plt.subplots(figsize=(8, 6))
plt.title('Hateful Sentiment vs Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
ax.plot(grouped.index, grouped)









