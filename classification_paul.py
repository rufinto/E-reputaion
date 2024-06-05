from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import test
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
from nltk.tag import StanfordPOSTagger

jar = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
model = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/models/french-ud.tagger'
import os
java_path = "C:/Program Files (x86)/Java/jre-1.8/bin/java.exe"
os.environ['JAVAHOME'] = java_path
pos_tagger_1 = StanfordPOSTagger(model, jar, encoding='utf8')


tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())


# Supposons que X contient vos tweets et y contient vos étiquettes
# X et y sont des listes de même longueur, X contient les tweets et y contient les étiquettes continues entre -1 et 1
X=[test.filtered_tweets[i] for i in range(len(test.filtered_tweets))]
analysis = [tb(tweet) for tweet in X]


Y =[analysis.sentiment[0] for analysis in analysis]
# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convertissez les tweets en représentation TF-IDF
vectorizer = TfidfVectorizer()


# Convertir les tweets en représentation TF-IDF
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialisez et entraînez le modèle RandomForest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_tfidf, y_train)

# Faites des prédictions sur l'ensemble de test
y_pred = rf_regressor.predict(X_test_tfidf)

# Évaluez les performances du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Vous pouvez également utiliser le modèle entraîné pour prédire les étiquettes des nouveaux tweets
# new_tweets_tfidf = vectorizer.transform(new_tweets)
# new_tweets_sentiments = rf_regressor.predict(new_tweets_tfidf)

from sklearn.metrics import recall_score

# Convertir les valeurs continues en catégories
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
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs. Valeurs Réelles')
plt.show()

# Calculer le rappel
recall_pos= recall_score(y_test_binned, y_pred_binned, pos_label=1)
recall_neg = recall_score(y_test_binned, y_pred_binned, pos_label=0)
print(f'Rappel pour les tweets positifs: {recall_pos:.2f}')
print(f'Rappel pour les tweets négatifs: {recall_neg:.2f}')