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
#Tokenization + creation corpus de mots
    

list_tweets = [word_tokenize(tweet) for tweet in X_set] #liste de tweets tokenizes

#####################################################################################
#filtration

def filtrage(tweet):
    stop_words = set(stopwords.words("french"))
    mots_utiles = {"ne", "pas", "n"} # stop word à conserver
    mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", "|", ".", "https", "http", "/", "``","&"} #stop word supplémentaires

    stop_words = stop_words.difference(mots_utiles)
    stop_words = stop_words.union(mots_inutiles)

    return [word for word in tweet if word.casefold() not in stop_words]

filtered_list_tweets = [filtrage(tweet) for tweet in list_tweets]

#####################################################################################
#ici on retire les lien de site web dans les corpus qui ne sont pas utiles. Elle prend
#comme entrée la liste de token d'un tweet

def retire_site_web(tweet):
    for word in tweet:
        if word[0:2] == "//":
            tweet.remove(word)

for tweet in filtered_list_tweets:
    retire_site_web(tweet)

#####################################################################################
#lemmatisation

def lemmatizeur(tweet):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tweet]

lemmatized_list_tweets = [lemmatizeur(tweet) for tweet in filtered_list_tweets] # on lemmatise chaque tweet de la liste de tweets

#####################################################################################
#creation du sac de token filtré et lemmatizé

sac_de_mots = [] 
for tweet in lemmatized_list_tweets:
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
idf = compute_idf(lemmatized_list_tweets, tf_sac_de_mots)

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

def coordonnees_tweet(tweet):

    token = word_tokenize(tweet) #tokenization
    token = filtrage(token) #filtrage
    retire_site_web(token) #on supprime les liens de sites web
    token = lemmatizeur(token) #on lemmatise
    tf_token = FreqDist(token).most_common() #on trouve la liste des tfs

    coordonnees = [0]*taille_base
    for term in tf_token:
        if term[0] in base_mots: #on peut ameliorer la complexite avec un dictionnaire
            coordonnees[base_mots.index(term[0])] = term[1]
    return coordonnees

#####################################################################################
#implémentation KNN

K = 5
KNN_model = sklearn.model_selection.knn
