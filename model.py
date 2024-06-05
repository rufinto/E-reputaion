import pandas as pd
import sklearn
import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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

'''def print_tweet(X, debut, fin):
    for i in range(debut-1, fin):
        print(f"{i+1}) {X[i]}\n\n")'''


"""on a split le dataset en trois: 10% pour le dictionnaire de groupe de mots, 70% pour le train et 20% pour le test"""

corpus_2= []
for tweet in dataset[columns[1]]:
    corpus_2.append(tweet)
# Tokenization et lemmatization
def lemmatize_tweets(tweets):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("french"))
    mots_utiles = {"ne", "pas", "n"} 
    mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", ".", "https", "http", "/", "``","&"} 

    stop_words = stop_words.difference(mots_utiles)
    stop_words = stop_words.union(mots_inutiles)

    lemmatized_tweets = []
    for tweet in tweets:
        words = word_tokenize(tweet)
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        lemmatized_tweets.append(' '.join(lemmatized_words))
    
    return lemmatized_tweets

# Fonction pour calculer les scores TF-IDF
tweets_lemmatizes=lemmatize_tweets(corpus_2)
def calculate_tfidf_for_tweets(tweets_lemmatizes):
    # Initialisation du vectorizer
    vectorizer = TfidfVectorizer()
    # Calcul des scores TF-IDF
    tfidf_matrix = vectorizer.fit_transform(tweets_lemmatizes)
    # Récupération des noms de fonction
    feature_names = vectorizer.get_feature_names_out()
    # Conversion de la matrice TF-IDF en tableau
    tfidf_scores = tfidf_matrix.toarray()
    # Récupération des mots des tweets lemmatisés
    lemmatized_words = [tweet.split() for tweet in tweets_lemmatizes]
    # Création du dictionnaire contenant les mots lemmatisés et leur score TF-IDF
    words_tfidf = {}
    for i, words in enumerate(lemmatized_words):
        for j, word in enumerate(words):
            if word not in words_tfidf:
                words_tfidf[word] = tfidf_scores[i][j]
            else:
                # Si le mot existe déjà, on prend le score TF-IDF maximum
                words_tfidf[word] = max(words_tfidf[word], tfidf_scores[i][j])
    # Tri du dictionnaire par ordre décroissant de score TF-IDF
    sorted_words_tfidf = sorted(words_tfidf.items(), key=lambda x: x[1], reverse=True)
    return sorted_words_tfidf
# Utilisation de la fonction pour calculer les scores TF-IDF pour les tweets lemmatisés
sorted_words_tfidf = calculate_tfidf_for_tweets(tweets_lemmatizes)
print(sorted_words_tfidf)
#####################################################################################
#calcul tf , renvoie une liste décroissante de mots selon leur tf

"""tf = FreqDist(lemmatized_sac_de_mots)
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
    tf_token = FreqDist(token).most_common() #on trouve la liste des tfs

    coordonnees = []
    for word in base_mot:
        coordonnees.append(0)
        for term in token:
            if term[0] == word:
                coordonnees[-1] = term[1]
                break
    return coordonnees"""