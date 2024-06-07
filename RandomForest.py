from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import test
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
from nltk.tag import StanfordPOSTagger
import pandas as pd
from nltk.tokenize import word_tokenize
import seaborn as sns
import KNRegressor
# Importing the dataset

dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name, sep = ";")

# Utilisez un autre POSTagger, inutile ici

jar = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
model = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/models/french-ud.tagger'
import os
java_path = "C:/Program Files (x86)/Java/jre-1.8/bin/java.exe"
os.environ['JAVAHOME'] = java_path
pos_tagger_1 = StanfordPOSTagger(model, jar, encoding='utf8')

#Labelling the tweets

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

# Filtration des tweets

mots_inutiles = {"\\", "\\n", "[", "]", "(", ")", "-", ":", ",", "#", "@", "»", "«", "''", "’", "'", "|", ".", "https", "http", "/", "``","&"}

def retire_site_web(liste):
    for word in liste:
        if word[0:2] == "//":
            liste.remove(word)

stop_words_1= mots_inutiles
filtered_tweets = {}
for i in range(len(dataset['data__text'])):
    tweet = dataset['data__text'][i]
    tweet = word_tokenize(tweet)
    tweet = [word for word in tweet if word.casefold() not in stop_words_1]
    tweet = ' '.join(tweet)
    
    filtered_tweets[i]=tweet
    retire_site_web(filtered_tweets[i])

# X contient les tweets et Y les étiquettes continues entre -1 et 1
X=[filtered_tweets[i] for i in range(len(filtered_tweets))]
analysis = [tb(tweet) for tweet in X]


Y =[analysis.sentiment[0] for analysis in analysis]



# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = KNRegressor.X_train, KNRegressor.X_test, KNRegressor.Y_train, KNRegressor.Y_test

# Convertion des tweets en représentation TF-IDF
#vectorizer = TfidfVectorizer()



#X_train_tfidf = vectorizer.fit_transform(X_train)
#X_test_tfidf = vectorizer.transform(X_test)

# Initialisation et entraînement du modèle RandomForest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(KNRegressor.coordonnees_matrice(X_train), y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rf_regressor.predict(KNRegressor.coordonnees_matrice(X_test))

# Évaluation des performances du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# On peut également utiliser le modèle entraîné pour prédire les étiquettes des nouveaux tweets
# new_tweets_tfidf = vectorizer.transform(new_tweets)
# new_tweets_sentiments = rf_regressor.predict(new_tweets_tfidf)

from sklearn.metrics import recall_score

# Convertion des valeurs continues en catégories
# Positif (1) si la note est >= 0, négatif (0) si la note est < 0
y_test_binned = []
y_pred_binned = []
for y in y_test:
    if y >= 0:
        y_test_binned.append(1)
    else:
        y_test_binned.append(0)
for y in y_pred:
    if y >= 0:
        y_pred_binned.append(1)
    else:
        y_pred_binned.append(0)

import matplotlib.pyplot as plt

# Assumons que y_test contient les valeurs réelles et y_pred contient les prédictions du modèle
# Créer un DataFrame avec les valeurs réelles et prédites
data = pd.DataFrame({'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred})

plt.figure(figsize=(10, 6))
sns.violinplot(data=data, palette="Set3")
plt.title('Graphique en Violon des Valeurs Réelles et Prédites')
plt.show()
# Tracer les histogrammes
plt.figure(figsize=(12, 6))

# Histogramme des valeurs réelles
plt.hist(y_test, bins=30, alpha=0.6, label='Valeurs Réelles', color='blue', edgecolor='black')

# Histogramme des valeurs prédites
plt.hist(y_pred, bins=30, alpha=0.6, label='Valeurs Prédites', color='red', edgecolor='black')

# Ajouter des légendes et des titres
plt.xlabel('Valeurs')
plt.ylabel('Fréquence')
plt.title('Histogramme des Valeurs Réelles et Prédites')
plt.legend()

# Afficher le graphique
plt.show()


# Calculer le rappel
recall_pos= recall_score(y_test_binned, y_pred_binned, pos_label=1)
recall_neg = recall_score(y_test_binned, y_pred_binned, pos_label=0)
print(f'Rappel pour les tweets positifs: {recall_pos:.2f}')
print(f'Rappel pour les tweets négatifs: {recall_neg:.2f}')