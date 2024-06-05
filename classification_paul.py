from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Exemples de tweets et leurs étiquettes de sentiment
tweets = [
    "I love the new policies on immigration!",
    "The immigration system is broken and needs fixing.",
    "I'm not sure how I feel about the new immigration laws.",
    "The government is doing a great job with immigration.",
    "The new laws are terrible for immigrants.",
    "I think the new policies are fine.",
    "Immigration reform is needed urgently.",
    "I'm happy with the changes to immigration laws."
]
labels = ["positive", "negative", "neutral", "positive", "negative", "neutral", "negative", "positive"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.3, random_state=42)

# Créer un pipeline avec TfidfVectorizer et RandomForestClassifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Afficher les résultats
print(classification_report(y_test, y_pred))

# Exemple de prédiction avec de nouveaux tweets
new_tweets = [
    "The new immigration policies are great!",
    "I hate the changes to the immigration laws."
]
predicted_labels = model.predict(new_tweets)

for tweet, label in zip(new_tweets, predicted_labels):
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {label}")
