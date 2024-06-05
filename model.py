import pandas as pd
import sklearn
import numpy as np 

import nltk
import sklearn.model_selection
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
Y = [0.8, 0, 0.6, 0, 0, 0, 1, 0.5, 1, -0.3, 0.5, -1, 0, -0.5, -0.5, 0 , 1, -0.3, 0, -1, 0, -0.3, -1, -0.5, -1, 0.5,
     -0.5, -1, -0.1, 0, -0.5, 0.5, -0.2, 1, 0.5, -0.1, 0, -0.5, -0.5, -0.5, 0, 1, -0.2, -0.5, 0.2, -1, 0, 0.5, -0.5, 
     -1, 1, 1, 0, -0.5, 0, -0.1, 0, -1, 1, 0.5, 0.5, -0.2, -0.1, -0.5, -0.3, -0.1, -0.1, -0.2, -0.2, -0.2, -0.5, -0.2,
     -0.2, 0, 0.5, -0.5, -1, -0.2, 0.5, -0.3, -0.1, 0, 0, 1, 0.5, 0, -1, -0.5, 0, 0.5, 0, 0, 0.5, -1, 0.5, 1, -0.5, -1,
     -1, 0.5, -0.3, 0, -0.4, -0.3, -0.1, -1, -0.5, -0.6, 0.8, -0.6, -0.5, -0.4, -0.5, -0.6, -0.9, -0.2, -0.6, -0.8, 0,
     -1, -0.5, -0.8,-0.6, -0.5, -0.8, -0.4, -0.6, -0.7, -0.9, 0, 0, -0.8, -0.4, 0.8, 0, 0.3, 0.2,-0.3, -0.1, -0.5,
     0.7, -1, -0.6, 0, 0.4, 0.8, -0.6, -0.9, 0, -0.4, -0.5, -0.5, 0, 0, -0.5, 0, 0, 0.5, -0.4, 0, -0.5, -0.8, 0, -1,
     -0.8, -0.3, -0.8, 0.5, 0.3, -0.9, 0, -0.5, -0.7, -0.9, 0, -0.5, 0, 0,-0.4, 0.9, -0.9, 0, -0.3, 0.7, 0.5, -0.4, 0,
     -0.4, -0.3, 0.6, -0.8, -0.5,-0.7, 0, 0.5, -0.6, -0.4, -0.4, -0.8,0, 0.15, -0.90, 0, -0.60, 0, -0.2, 0.3, -0.2, -0.6,
     -0.4, 0.6, -0.8, -0.35, -0.4, -0.1, -0.35, -0.7, 0, -0.15, 0.6, 0.6, 0, -0.4, -0.2, 0, 0.6, -0.5, 0.4, 0.23, 0.9,
     -0.2, -0.6, 0.1, -0.1, -0.7, 0, -0.3, 0, 0.8, 0.1, -0.2, -0.4, 0.6, 0, 0.34, -0.6, 0.3, -0.15, 0.9, 0.4, -0.4, -0.4,
     -0.1, 0, 0.2, -0.5, 0.45, -0.3, 0.1, 0.2, -0.51, -0.5, -0.2, -0.5, -0.2, -0.48, -0.3, 0.48, -0.1, 0, 0.3, -0.3, 0.1,
     -0.3, -0.2, 0.6, 0.4, 0.3, 0.5, 0.7, 0, 0.45, 0.32, 0.2, -0.4, -0.45, 0, -0.1, 0.67, 0.8, -0.6, -0.3, -0.2, 0, 0.71,
     0.6, 0.67, -0.4, 0.2, -0.2]

def print_tweet(X, debut, fin):
    for i in range(debut-1, fin):
        print(f"{i+1}) {X[i]}\n\n")


"""on a split le dataset en trois: 10% pour le dictionnaire de groupe de mots, 70% pour le train et 20% pour le test"""

X_set, X_test, Y_set, Y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.2, random_state=100)

X_train, Y_train, X_sac_mots, Y_sac_mots = sklearn.model_selection.train_test_split(
    X_set, Y_set, test_size=0.125, random_state=100) #10% du train initail c'est 12,5% du X_set qui lui même vaut 80% du train initial



#####################################################################################
#creation corpus


corpus = ""
for tweet in X_set:
    corpus += tweet


#####################################################################################
#Tokenization
    

sac_de_mots = word_tokenize(corpus)

#####################################################################################
#filtration

def filtrage(sac_de_mots):
    stop_words = set(stopwords.words("french"))
    mots_utiles = {"ne", "pas", "n"} # stop word à conserver
    mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", ".", "https", "http", "/", "``","&"} #stop word supplémentaires

    stop_words = stop_words.difference(mots_utiles)
    stop_words = stop_words.union(mots_inutiles)

    return [word for word in sac_de_mots if word.casefold() not in stop_words]

filtered_sac_de_mots = filtrage(sac_de_mots)

#####################################################################################
#ici on retire les lien de site web dans les corpus qui ne sont pas utiles

def retire_site_web(liste):
    for word in liste:
        if word[0:2] == "//":
            liste.remove(word)

retire_site_web(filtered_sac_de_mots)

#####################################################################################
#lemmatisation

def lemmatizeur(filtered_sac_de_mots):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, ) for word in filtered_sac_de_mots]

lemmatized_sac_de_mots = lemmatizeur(filtered_sac_de_mots)

#####################################################################################
#calcul tf , renvoie une liste décroissante de mots selon leur tf

tf = FreqDist(lemmatized_sac_de_mots)
tf.plot(20, cumulative=True)

#####################################################################################
#base de mot

base_mot = [val[0] for val in tf.most_common()]

#####################################################################################
#representation d'un vecteur dans la base

def coordonnees_tweet(tweet):

    token = word_tokenize(tweet) #tokenization
    token = filtrage(token) #filtrage
    token = retire_site_web(token) #on supprime les liens de sites web
    token = lemmatizeur(token) #on lemmatise
    tf_token = FreqDist(token) #on trouve la liste des tfs

    coordonnees = []

