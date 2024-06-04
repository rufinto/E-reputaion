import pandas as pd
import sklearn

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name, sep=";")
colums = ["data__id", "data__text", "data__created_at", "author__username"]
nbr_tweets = len(dataset[colums[0]])

#trainset = sklearn.
#print(nbr_tweets)

#creation corpus
corpus = ""
for tweet in dataset[colums[1]]:
    corpus += tweet


#Tokenization

sac_de_mots = word_tokenize(corpus)

#filtration

stop_words = set(stopwords.words("french"))
mots_utiles = {"ne", "pas", "n"} # stop word à conserver
mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", ".", "https", "http", "/", "``","&"} #stop word supplémentaires

stop_words = stop_words.difference(mots_utiles)
stop_words = stop_words.union(mots_inutiles)

filtered_sac_de_mots = [word for word in sac_de_mots if word.casefold() not in stop_words]

def retire_site_web(liste):
    for word in liste:
        if word[0:2] == "//":
            liste.remove(word)

retire_site_web(filtered_sac_de_mots)
#print(filtered_sac_de_mot, len(filtered_sac_de_mot))

#lemmatisation
lemmatizer = WordNetLemmatizer()
lemmatized_sac_de_mots = [lemmatizer.lemmatize(word) for word in filtered_sac_de_mots]

#calcul tf

tf = FreqDist(lemmatized_sac_de_mots)
print(tf.most_common(50), len(tf))

