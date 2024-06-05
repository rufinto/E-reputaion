import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk 


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# Importing the dataset
chemin = 'FR-L-MIGR-TWIT-2011-2022.csv'
dataset = pd.read_csv(chemin, sep=';')

# Corpus creation 
corpus = ""
for tweet in dataset['data__text']:
    corpus += tweet


#Tokenization

sac_de_mots = word_tokenize(corpus)

# Filter the dataset


stop_words = set(stopwords.words("french"))
mots_utiles = {"ne", "pas", "n"} # stop word à conserver
mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", ".", "https", "http", "/", "``","&"} #stop word supplémentaires



stop_words = stop_words.difference(mots_utiles)
stop_words = stop_words.union(mots_inutiles)
def retire_site_web(liste):
    for word in liste:
        if word[0:2] == "//":
            liste.remove(word)

stop_words_1= mots_inutiles
filtered_tweets = {}
for i in range(len(dataset['data__text'])):
    tweet = dataset['data__text'][i]
    tweet = word_tokenize(tweet)
    tweet = [word for word in tweet if word.casefold() not in stop_words_1]
    tweet = ' '.join(tweet)
    
    filtered_tweets[i]=tweet
    retire_site_web(filtered_tweets[i])

filtered_tweets_by_word = {}
for i in range(len(dataset['data__text'])):
    tweet1 = dataset['data__text'][i]
    tweet1 = word_tokenize(tweet1)
    tweet1 = [word for word in tweet if word.casefold() not in stop_words_1]
    
    
    filtered_tweets_by_word[i]=tweet1
    retire_site_web(filtered_tweets_by_word[i])

