import pandas as pd
import numpy as np
import nltk
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

# Utilisez le chemin absolu vers le fichier CSV
dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name, sep = ";")

columns = ["data__id", "data__text", "data__created_at", "author__username"]
dataset_length = len(dataset[columns[0]])

# Sélection aléatoire des tweets
n = 300
np.random.seed(100)
tweets_Id = np.random.choice(dataset_length, n, replace=False)
X = [dataset[columns[1]][i] for i in tweets_Id]
Y = [0.8, 0, 0.6, 0, 0, 0, 1, 0.5, 1, -0.3, 0.5, -1, 0, -0.5, -0.5, 0, 1, -0.3, 0, -1, 0, -0.3, -1, -0.5, -1, 0.5,
     -0.5, -1, -0.1, 0, -0.5, 0.5, -0.2, 1, 0.5, -0.1, 0, 0, 1, 0.5, 0, -1, -0.5, 0, 0.5, 0, 0, 0.5, -1, 0.5, 1, -0.5, -1,
     -1, 0.5, -0.3, 0, -0.4, -0.3, -0.1, -1, -0.5, -0.6, 0.8, -0.6, -0.5, -0.4, -0.5, -0.6, -0.9, -0.2, -0.6, -0.8, 0,
     -1, -0.5, -0.8, -0.6, -0.5, -0.8, -0.4, -0.6, -0.7, -0.9, 0, 0, -0.8, -0.4, 0.8, 0, 0.3, 0.2, -0.3, -0.1, -0.5,
     0.7, -1, -0.6, 0, 0.4, 0.8, -0.6, -0.9, 0, -0.4, -0.5, -0.5, 0, 0, -0.5, 0, 0, 0.5, -0.4, 0, -0.5, -0.8, 0, -1,
     -0.8, -0.3, -0.8, 0.5, 0.3, -0.9, 0, -0.5, -0.7, -0.9, 0, -0.5, 0, 0, -0.4, 0.9, -0.9, 0, -0.3, 0.7, 0.5, -0.4, 0,
     -0.4, -0.3, 0.6, -0.8, -0.5, -0.7, 0, 0.5, -0.6, -0.4, -0.4, -0.8, 0, 0.15, -0.90, 0, -0.60, 0, -0.2, 0.3, -0.2, -0.6,
     -0.4, 0.6, -0.8, -0.35, -0.4, -0.1, -0.35, -0.7, 0, -0.15, 0.6, 0.6, 0, -0.4, -0.2, 0, 0.6, -0.5, 0.4, 0.23, 0.9,
     -0.2, -0.6, 0.1, -0.1, -0.7, 0, -0.3, 0, 0.8, 0.1, -0.2, -0.4, 0.6, 0, 0.34, -0.6, 0.3, -0.15, 0.9, 0.4, -0.4, -0.4,
     -0.1, 0, 0.2, -0.5, 0.45, -0.3, 0.1, 0.2, -0.51, -0.5, -0.2, -0.5, -0.2, -0.48, -0.3, 0.48, -0.1, 0, 0.3, -0.3, 0.1,
     -0.3, -0.2, 0.6, 0.4, 0.3, 0.5, 0.7, 0, 0.45, 0.32, 0.2, -0.4, -0.45, 0, -0.1, 0.67, 0.8, -0.6, -0.3, -0.2, 0, 0.71,
     0.6, 0.67, -0.4, 0.2, -0.2]

# Split du dataset en trois
X_set, X_test, Y_set, Y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.2, random_state=100)

X_train, X_sac_mots, Y_train, Y_sac_mots = sklearn.model_selection.train_test_split(
    X_set, Y_set, test_size=0.125, random_state=100) # 12,5% de X_set = 10% du train initial

# Tokenization et filtration
def filtrage(sac_de_mots):
    stop_words = set(stopwords.words("french"))
    mots_utiles = {"ne", "pas", "n"}
    mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", ".", "https", "http", "/", "``","&"}
    stop_words = stop_words.difference(mots_utiles)
    stop_words = stop_words.union(mots_inutiles)
    return [word for word in sac_de_mots if word.casefold() not in stop_words]

# Lemmatisation
def lemmatizeur(filtered_sac_de_mots):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in filtered_sac_de_mots]

# Construction du corpus de base
corpus = " ".join(X_train)
sac_de_mots = word_tokenize(corpus)
filtered_sac_de_mots = filtrage(sac_de_mots)
filtered_sac_de_mots = [word for word in filtered_sac_de_mots if not word.startswith("http")]
lemmatized_sac_de_mots = lemmatizeur(filtered_sac_de_mots)

# Calcul du TF
tf = FreqDist(lemmatized_sac_de_mots)
base_mot = [val[0] for val in tf.most_common()]

# Création du TF-IDF
vectorizer = TfidfVectorizer(vocabulary=base_mot)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train_tfidf, Y_train)

# Prédictions sur le jeu de test
Y_pred = model.predict(X_test_tfidf)

# Affichage des résultats
for i in range(len(X_test)):
    print(f"Tweet: {X_test[i]}")
    print(f"Vrai score: {Y_test[i]}")