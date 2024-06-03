import pandas as pd
import sklearn

import nltk
nltk.download('punkt')
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

dataset_name = "FR_L_MIGR_TWIT_2011_2022.csv"
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
#print(len(sac_de_mots))

#filtration

stop_words = set(stopwords.words("french"))
mots_utiles = {"ne", "pas", "n"} # stop word à conserver
mots_inutiles = {"\\", "\\n", "[", "]", ":", ",", "#", "@", "»", "«", "''", ".", "https", "/"} #stop word supplémentaires

stop_words = stop_words.difference(mots_utiles)
stop_words = stop_words.union(mots_inutiles)

print(stop_words)
filtered_sac_de_mot = [word for word in sac_de_mots if word.casefold() not in stop_words]

def retire_site_web(liste):
    for word in liste:
        if word[0:2] == "//":
            liste.remove(word)

retire_site_web(filtered_sac_de_mot)
print(filtered_sac_de_mot, len(filtered_sac_de_mot))