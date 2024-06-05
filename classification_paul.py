from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Exemples de tweets et leurs étiquettes de sentiment
tweets = [
    "I love the new policies on immigration!",
    "The immigration system is broken and needs fixing.",
    "I'm not sure how I feel about the new immigration laws."
]
labels = ["positive", "negative", "neutral"]

# Créer un pipeline avec CountVectorizer et MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entraîner le modèle
model.fit(tweets, labels)

# Faire des prédictions
new_tweets = [
    "The new immigration policies are great!",
    "I hate the changes to the immigration laws."
]
predicted_labels = model.predict(new_tweets)

# Afficher les résultats
for tweet, label in zip(new_tweets, predicted_labels):
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {label}")
