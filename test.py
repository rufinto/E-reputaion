import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk 
import flair

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# Importing the dataset
chemin = 'FR-L-MIGR-TWIT-2011-2022.csv'
dataset = pd.read_csv(chemin, sep=';')



#for i in dataset['data__text'].head():
    #print(word_tokenize(i))

french_stopwords = set(stopwords.words('french'))
stemmer = nltk.stem.SnowballStemmer('french')
print(french_stopwords)
for i in dataset['data__text'].head():
    print([stemmer.stem(word) for word in word_tokenize(i) if word.lower() not in french_stopwords])
    #print(word_tokenize(i))
    #print([word for word in word_tokenize(i) if word.lower() not in french_stopwords])