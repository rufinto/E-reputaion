import pandas as pd
import sklearn
import numpy as np 

import nltk
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

#####################################################################################
#lecture du dataset et etiquetage

dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name, sep = ";")
columns = ["data__id", "data__text", "data__created_at", "author__username"]
dataset_lenght = len(dataset[columns[0]]) #donne le nombre de tweets dans le dataset


n = 300
#mettre 1 quand le tweet est postif et 0 quand il est négatif
np.random.seed(100) #permet de fixer toutes expériences aléatoires éffectuées avec random afin que le résultat reste le même afin chaque exécution
tweets_Id = np.random.choice(dataset_lenght, n, replace=False)
X = [dataset[columns[1]][i] for i in tweets_Id]
Y = ["1 à 100",
     "101 à 200",
     "201 à 300"]

def print_tweet(X, debut, fin):
    for i in range(debut-1, fin):
        print(f"{i+1}) {X[i]}\n\n")

print_tweet(X, debut=101, fin=200)

#####################################################################################
#creation corpus

"""
corpus = ""
for tweet in dataset[colums[1]]:
    corpus += tweet

"""
#####################################################################################
#Tokenization
    
"""
sac_de_mots = word_tokenize(corpus)
"""

#####################################################################################
#filtration

"""
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
"""

#####################################################################################
#lemmatisation

"""
lemmatizer = WordNetLemmatizer()
lemmatized_sac_de_mots = [lemmatizer.lemmatize(word, ) for word in filtered_sac_de_mots]
"""

#####################################################################################
#calcul tf

"""
tf = FreqDist(lemmatized_sac_de_mots)
print(tf.most_common(50), len(tf))
"""