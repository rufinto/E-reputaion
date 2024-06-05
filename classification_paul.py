from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import test
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer


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
