import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from Preprocessing import X_train, X_test, Y_train, Y_test
from Preprocessing import coordonnees_matrice
from Preprocessing import coordonnees_tweet, nettoyage_tweet
import sklearn
import sklearn.model_selection
import sklearn.neighbors


# Définition du nombre de voisins pour le modèle KNeighborsRegressor
K = 5

# Initialisation et entraînement du modèle KNeighborsRegressor
KNR_model = KNeighborsRegressor(n_neighbors=K)
KNR_model.fit(coordonnees_matrice(X_train), Y_train)

def accuracy():

    """accuracy_type = 0 signifie qu'on regarde juste si c'est positif / négatif / neutre par rapport à la prédiction
       avec trois classes : [-1, 0.05[, [-0.5, 0.5], ]0.05, 1]
       accuracy_type = 1 si on calcule l'erreur quadratique moyenne de la prédiction par rapport à la cible""" 

    # Extraction des valeurs cibles (vraies étiquettes) à partir des données de test
    targets = np.array(Y_test)
    predictions = []

    for tweet in X_test:
        tweet_coordonnees = np.array(coordonnees_tweet(nettoyage_tweet(tweet))).reshape(1, -1)
        predictions.append(KNR_model.predict(tweet_coordonnees))
    predictions = np.array(predictions)

    # Calcul de l'erreur quadratique moyenne pour évaluer la précision du modèle
    erreur = (predictions - targets)**2
    erreur = erreur.mean()
    print(f"Erreur quadratique moyenne = {erreur}")
    
    # Calcul du nombre de prédictions correctes par rapport aux vraies étiquettes pour évaluer la précision du modèle
    prediction_vraies = 0
    for score, tag in zip(predictions, targets):
        if score < -0.05 and tag < 0:
            prediction_vraies += 1
        elif score >= -0.05 and score <= 0.05 and tag == 0 : 
            prediction_vraies += 1
        elif score > 0.05 and tag > 0:
            prediction_vraies += 1
    accuracy_rate = prediction_vraies / len(targets)
    print(f"Pourcentage de réussite = {accuracy_rate*100}%")

accuracy()
