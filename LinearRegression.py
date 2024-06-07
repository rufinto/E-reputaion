import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve 
from Preprocessing import X_train, X_test, Y_train, Y_test
from Preprocessing import coordonnees_matrice,coordonnees_tweet, nettoyage_tweet

X_train_final=coordonnees_matrice(X_train)

regressor = LinearRegression()

# Entraînez le modèle sur les données d'entraînement
regressor.fit(X_train_final, Y_train)

# Prédisez les valeurs sur les données de test
predictions = []
for tweet in X_test:
        tweet_coordonnees = np.array(coordonnees_tweet(nettoyage_tweet(tweet))).reshape(1, -1)
        predictions.append(regressor.predict(tweet_coordonnees))
predictions = np.array(predictions) 

Y_pred = predictions
# Évaluez les performances du modèle
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Fonction pour classifier les valeurs en positif, neutre et négatif
def classify(y):
    y_classified = []
    for value in y:
        if value > 0.05:
            y_classified.append(1)  # Positif
        elif value < -0.05:
            y_classified.append(-1)  # Négatif
        else:
            y_classified.append(0)  # Neutre
    return y_classified

# Classifier les prédictions et les valeurs réelles
Y_pred_classified = classify(Y_pred)
Y_test_classified = classify(Y_test)

# Calculer le rappel pour chaque classe
recall_pos = recall_score(Y_test_classified, Y_pred_classified, labels=[1], average='macro')
recall_neutre = recall_score(Y_test_classified, Y_pred_classified, labels=[0], average='macro')
recall_neg = recall_score(Y_test_classified, Y_pred_classified, labels=[-1], average='macro')

print(f"Rappel pour la classe positive: {recall_pos}")
print(f"Rappel pour la classe neutre: {recall_neutre}")
print(f"Rappel pour la classe négative: {recall_neg}")
print("Erreur Quadratique Moyenne (MSE):", mse)
print("Coefficient de Détermination (R-squared):", r2)

# Tracer l'histogramme des prédictions
plt.figure(figsize=(10, 5))
plt.hist(Y_pred, bins=20, color='blue', alpha=0.7)
plt.title('Histogramme des Prédictions')
plt.xlabel('Valeurs Prédites')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# Tracer l'histogramme des valeurs réelles (Y_test)
plt.figure(figsize=(10, 5))
plt.hist(Y_test, bins=20, color='green', alpha=0.7)
plt.title('Histogramme des Valeurs Réelles (Y_test)')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# Matrice de Confusion
conf_matrix = confusion_matrix(Y_test_classified, Y_pred_classified)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.title('Matrice de Confusion')
plt.show()

# Binariser les étiquettes pour la courbe ROC et l'AUC
Y_test_binarized = label_binarize(Y_test_classified, classes=[-1, 0, 1])
Y_pred_binarized = label_binarize(Y_pred_classified, classes=[-1, 0, 1])

# Calculer la courbe ROC et l'AUC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(Y_test_binarized[:, i], Y_pred_binarized[:, i])
    roc_auc[i] = roc_auc_score(Y_test_binarized[:, i], Y_pred_binarized[:, i])

# Tracer toutes les courbes ROC
plt.figure()
colors = ['blue', 'green', 'red']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i))
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# Courbe de Précision-Rappel
precision = dict()
recall = dict()
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(Y_test_binarized[:, i], Y_pred_binarized[:, i])

# Tracer toutes les courbes Precision-Recall
plt.figure()
for i, color in zip(range(3), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-Recall curve for class %i' % i)
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe Precision-Recall')
plt.legend(loc="lower left")
plt.show()

# Graphique de la distribution des erreurs
errors = Y_test - Y_pred

plt.hist(errors, bins=20)
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.title('Distribution des Erreurs')
plt.show()

# Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(regressor, X_train, Y_train, cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score d\'entraînement')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Score de validation')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

plt.xlabel('Taille de l\'ensemble d\'entraînement')
plt.ylabel('Score')
plt.title('Courbe d\'apprentissage')
plt.legend(loc='best')
plt.show()

# Afficher les coefficients du modèle
coefficients = regressor.coef_
intercept = regressor.intercept_

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

