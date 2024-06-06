import pandas as pd
import sklearn
import numpy as np 

import nltk
import sklearn
import sklearn.model_selection
import sklearn.neighbors
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib.pyplot as plt

"""
pour lancer le model il faut aller à la ligne 132 pour choisir si l'on veut l'entraîner sur le dataset etiqueté manuellement
et contenants 300 tweets (choisir dtyp=0), ou sur le dataset etiqueté automatiquement et qui contient tous les tweets (dtyp=1).
pour le datase complet ça prendra un peu de temps car il ya un corpus de plus de 100k mots à traiter, il faut environ 5 minutes
delon la puissance de la machine avec laquelle vous exécuter le code. 

"""

#####################################################################################
#lecture du dataset et etiquetage

dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name, sep = ";")
columns = ["data__id", "data__text", "data__created_at", "author__username"]
dataset_lenght = len(dataset[columns[0]]) #donne le nombre de tweets dans le dataset

#####################################################################################
#etuiquetage manuel de n tweets
    
n = 300
#mettre 1 quand le tweet est postif et 0 quand il est négatif
np.random.seed(100) #permet de fixer toutes expériences aléatoires éffectuées avec random afin que le résultat reste le même afin chaque exécution
tweets_Id = np.random.choice(dataset_lenght, n, replace=False)

X_manuel = [dataset[columns[1]][i] for i in tweets_Id]
Y_manuel = [0.8, 0, 0.6, 0, 0, 0, 1, 0.5, 1, -0.3, 0.5, -1, 0, -0.5, -0.5, 0 , 1, -0.3, 0, -1, 0, -0.3, -1, -0.5, -1, 0.5,
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

# Divisez les données en ensembles d'entraînement et de test
"""on a split le dataset en trois: 80% pour le train et 20% pour le test"""

X_train_manuel, X_test_manuel, Y_train_manuel, Y_test_manuel = sklearn.model_selection.train_test_split(
    X_manuel, Y_manuel, test_size=0.2, random_state=100)

#####################################################################################
#filtration

def filtrage(tweet):
    stop_words = set(stopwords.words("french"))
    mots_utiles = {"ne", "pas", "n"} # stop word à conserver
    mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", "|", ".", "https", "http", "/", "``","&"} #stop word supplémentaires

    stop_words = stop_words.difference(mots_utiles)
    stop_words = stop_words.union(mots_inutiles)

    return [word for word in tweet if word.casefold() not in stop_words]

#####################################################################################
#ici on retire les lien de site web dans les corpus qui ne sont pas utiles. Elle prend
#comme entrée la liste de token d'un tweet

def retire_site_web(tweet):
    for word in tweet:
        if word[0:2] == "//":
            tweet.remove(word)

#####################################################################################
#etiquettage automatique de tous les tweets
    
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

# Supposons que X contient vos tweets et y contient vos étiquettes
# X et y sont des listes de même longueur, X contient les tweets et y contient les étiquettes continues entre -1 et 1

X_all = [tweet for tweet in dataset[columns[1]]]
X_all = [word_tokenize(tweet) for tweet in X_all]
X_all = [filtrage(tweet) for tweet in X_all]
for tweet in X_all:
    retire_site_web(tweet)
X_automatique = [" ".join(tweet) for tweet in X_all]
Analysis = [tb(tweet) for tweet in X_automatique]
Y_automatique = [analysis.sentiment[0] for analysis in Analysis]

X_train_automatique, X_test_automatique, Y_train_automatique, Y_test_automatique = sklearn.model_selection.train_test_split(X_automatique, Y_automatique, test_size=0.2)
#####################################################################################
#lemmatisation

def lemmatizeur(tweet):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tweet]

def Datasets(dtyp):

    """renvoie la liste de tweets tokenizée, filtrée, lemmatizée.
    dtyp = 0 Si dataset manuel
    dtyp = 1 si dataset etiquuetée numériquement"""
    #####################################################################################
    #Tokenization + creation corpus de mots
    
    if dtyp == 0:
        X_train, X_test, Y_train, Y_test = X_train_manuel, X_test_manuel, Y_train_manuel, Y_test_manuel
    elif dtyp == 1:
        X_train, X_test, Y_train, Y_test = X_train_automatique, X_test_automatique, Y_train_automatique, Y_test_automatique
    list_tweets = [word_tokenize(tweet) for tweet in X_train] #liste de tweets tokenizes
    filtered_list_tweets = [filtrage(tweet) for tweet in list_tweets]
    for tweet in filtered_list_tweets:
        retire_site_web(tweet)
    lemmatized_list_tweets = [lemmatizeur(tweet) for tweet in filtered_list_tweets] # on lemmatise chaque tweet de la liste de tweets
    return list_tweets, X_test, Y_train, Y_test

#####################################################################################
#creation du sac de token filtré et lemmatizé
X_train, X_test, Y_train, Y_test = Datasets(dtyp = 0)

sac_de_mots = [] 
for tweet in X_train:
    sac_de_mots += tweet

#on calcul la tf
tf = FreqDist(sac_de_mots)
tf_sac_de_mots = [word[0] for word in tf.most_common()] #le sac de mots obtenu en éliminant les doublons

#calcul tf-id, renvoie une liste décroissante de mots selon leur tf-id
def compute_idf(list_tweets, sac_de_mots):
    num_documents = len(list_tweets)
    idf = []
    
    for word in sac_de_mots:
        # Compter le nombre de documents contenant le mot
        num_documents_containing_word = sum(1 for doc in list_tweets if word in doc)
        
        # Calculer l'IDF pour ce mot
        idf_value = np.log(num_documents / (num_documents_containing_word))   # Ajouter 1 pour éviter les divisions par zéro
        idf.append((word, idf_value))
    return idf

#on calcul l'idf
idf = compute_idf(X_train, tf_sac_de_mots)

#on calcul tf-idf
def compute_tf_idf(tf_list, idf_list):

    tf_idf_list = []
    for word_tf, word_idf in zip(tf_list, idf_list):
        if word_tf[0] != word_idf[0]:
            raise ValueError("Les listes de tuples ne sont pas alignées.")
        tf_idf_list.append((word_tf[0], word_tf[1] * word_idf[1]))

    return tf_idf_list

tf_idf = compute_tf_idf(tf.most_common(), idf)

#tris par odre décroissant par rapport à au tf-idf
tf_idf_value = lambda sample: sample[1] #fonction qui va permettre de faire le tri par rapport à la la valeur de tf-idf
sorted(tf_idf, key=tf_idf_value)

#####################################################################################
#base de mot
base_mots = [val[0] for val in tf_idf]
taille_base = len(base_mots)

#####################################################################################
#representation d'un vecteur dans la base

def nettoyage_tweet(tweet):

    token = word_tokenize(tweet) #tokenization
    token = filtrage(token) #filtrage
    retire_site_web(token) #on supprime les liens de sites web
    token = lemmatizeur(token) #on lemmatise
    tf_token = FreqDist(token).most_common() #on trouve la liste des tfs
    return tf_token


def coordonnees_tweet(tf_token):

    coordonnees = [0]*taille_base
    for term in tf_token:
        if term[0] in base_mots: #on peut ameliorer la complexite avec un dictionnaire
            coordonnees[base_mots.index(term[0])] = term[1]
    return coordonnees

#####################################################################################
#implémentation KNN

#fonction de vectorisation
def coordonnees_matrice(X):
    return np.array([coordonnees_tweet(FreqDist(tweet).most_common()) for tweet in X])

K = 5
KNR_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=K)
KNR_model.fit(coordonnees_matrice(X_train), Y_train)

def accuracy():

    """accuracy_type = 0 signifie qu'on regarde juste si c'est positif / negatif / neutre par rapport à la prediction
       avec trois classe: [-1, 0.05[, [-0.5, 0.5], ]0.05, 1]
       accuracy_type = 1 si on calcule l'erreur quadratique moyenne de la prediction par rapport au traget""" 

    targets = np.array(Y_test)
    predictions = []

    for tweet in X_test:
        tweet_coordonnees = np.array(coordonnees_tweet(nettoyage_tweet(tweet))).reshape(1, -1)
        predictions.append(KNR_model.predict(tweet_coordonnees))
    predictions = np.array(predictions)

    erreur = (predictions - targets)**2
    erreur = erreur.mean()
    print(f"Erreur quadratiqe moyenne = {erreur}")
    
    prediction_vraies = 0
    for score, tag in zip(predictions, targets):
        if score < -0.05 and tag < 0:
            prediction_vraies += 1
        elif score >= -0.05 and score <= 0.05 and tag == 0 : 
            prediction_vraies += 1
        elif score > 0.05 and tag > 0:
            prediction_vraies += 1
    accuracy_rate = prediction_vraies / len(targets)
    print(f"Pourcetage de reussite = {accuracy_rate*100}%")

#accuracy()




