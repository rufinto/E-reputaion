from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import test
import matplotlib.pyplot as plt


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


# Créez une instance de régression linéaire
regressor = LinearRegression()

# Créez un pipeline avec le vectorizer et le modèle de régression linéaire
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('regression', regressor)
])

# Entraînez le modèle sur les données d'entraînement
pipeline.fit(X_train, y_train)

# Prédisez les étiquettes sur les données de test
y_pred = pipeline.predict(X_test)

# Évaluez les performances du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
# Tracer l'histogramme des prédictions
plt.figure(figsize=(10, 5))
plt.hist(y_pred, bins=20, color='blue', alpha=0.7)
plt.title('Histogramme des prédictions')
plt.xlabel('Valeurs prédites')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# Tracer l'histogramme des valeurs réelles (y_labellisé)
plt.figure(figsize=(10, 5))
plt.hist(y_test, bins=20, color='green', alpha=0.7)
plt.title('Histogramme des valeurs réelles (y labellisé)')
plt.xlabel('Valeurs réelles')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()
# Supposons que X contient vos tweets et y contient vos étiquettes
# X et y sont des listes de même longueur, X contient les tweets et y contient les étiquettes continues entre -1 et 1

