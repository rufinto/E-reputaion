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

dataset_Aubry = pd.read_csv("FR-L-Aubry-MIGR-TWIT-2022.csv", sep=";") #1
dataset_Autain = pd.read_csv("FR-L-Autain-MIGR-TWIT-2022.csv", sep=";") #2
dataset_Corbiere = pd.read_csv("FR-L-Corbière-MIGR-TWIT-2022.csv", sep=";") #3 
dataset_Glucksmann = pd.read_csv("FR-L-Glucksmann-MIGR-TWIT-2019.csv", sep=";") #4 
dataset_GRS = pd.read_csv("FR-L-GRS-MIGR-TWIT-2018.csv", sep=";") #5
dataset_Hollande = pd.read_csv("FR-L-Hollande-MIGR-TWIT-2021.csv", sep=";") #6 
dataset_Melonchon = pd.read_csv("FR-L-Mélenchon-MIGR-TWIT-2022.csv", sep=";") #7
dataset_Ruffin = pd.read_csv("FR-L-Ruffin-MIGR-TWIT-2021.csv", sep=";") #8
 
colums = ["data__id", "data__text", "data__created_at", "author__username"]

n = 50

X_Aubry = np.random.RandomState.choice([dataset_Aubry[colums[1]][i] for i in range(n)], n, replace=False)
Y_Aubry = []
np.random()
X_Autain = np.random.RandomState.choice([dataset_Autain[colums[1]][i] for i in range(n)], n, replace=False)
Y_Autain = []

X_Corbiere = np.random.RandomState.choice([dataset_Corbiere[colums[1]][i] for i in range(n)], n, replace=False)
Y_Corbiere = []

X_Glucksmann = np.random.RandomState.choice([dataset_Glucksmann[colums[1]][i] for i in range(n)], n, replace=False)
Y_Glucksmann = []

X_GRS = np.random.RandomState.choice([dataset_GRS[colums[1]][i] for i in range(n)], n, replace=False)
Y_GRS = []

X_Hollande = np.random.RandomState.choice([dataset_Hollande[colums[1]][i] for i in range(n)], n, replace=False)
Y_Hollande = []

X_Melonchon = np.random.RandomState.choice([dataset_Melonchon[colums[1]][i] for i in range(n)], n, replace=False)
Y_Melonchon = []

X_Ruffin = np.random.RandomState.choice([dataset_Ruffin[colums[1]][i] for i in range(n)], n, replace=False)
Y_Ruffin = []

print(X_Aubry.head())


#creation corpus

"""
corpus = ""
for tweet in dataset[colums[1]]:
    corpus += tweet

"""

#Tokenization
    
"""
sac_de_mots = word_tokenize(corpus)
"""

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

#lemmatisation

"""
lemmatizer = WordNetLemmatizer()
lemmatized_sac_de_mots = [lemmatizer.lemmatize(word, ) for word in filtered_sac_de_mots]
"""

#calcul tf

"""
tf = FreqDist(lemmatized_sac_de_mots)
print(tf.most_common(50), len(tf))
"""