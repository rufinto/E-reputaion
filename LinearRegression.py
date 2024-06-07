import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, recall_score, learning_curve
from sklearn.preprocessing import label_binarize







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

# Fonction pour classifier les valeurs en positif, neutre, négatif
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

# Classifier y_pred et y_test
y_pred_classified = classify(y_pred)
y_test_classified = classify(y_test)

# Calculer le rappel pour chaque classe
recall_pos = recall_score(y_test_classified, y_pred_classified, labels=[1], average='macro')
recall_neutre = recall_score(y_test_classified, y_pred_classified, labels=[0], average='macro')
recall_neg = recall_score(y_test_classified, y_pred_classified, labels=[-1], average='macro')

print(f"Rappel pour les classes positives: {recall_pos}")
print(f"Rappel pour les classes neutres: {recall_neutre}")
print(f"Rappel pour les classes négatives: {recall_neg}")
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

# Matrice de Confusion
conf_matrix = confusion_matrix(y_test_classified, y_pred_classified)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.title('Matrice de Confusion')
plt.show()

# Binariser les étiquettes pour la courbe ROC et AUC
y_test_binarized = label_binarize(y_test_classified, classes=[-1, 0, 1])
y_pred_binarized = label_binarize(y_pred_classified, classes=[-1, 0, 1])

# Calculer la courbe ROC et l'AUC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_pred_binarized[:, i])

# Tracer toutes les courbes ROC
plt.figure()
colors = ['blue', 'green', 'red']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i))
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Courbe Precision-Recall
precision = dict()
recall = dict()
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_binarized[:, i])

# Tracer toutes les courbes Precision-Recall
plt.figure()
for i, color in zip(range(3), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-Recall curve for class %i' % i)
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe Precision-Recall')
plt.legend(loc="lower left")
plt.show()

# Graphiques des Erreurs
errors = y_test - y_pred

plt.hist(errors, bins=20)
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.title('Distribution des Erreurs')
plt.show()

# Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train, y_train, cv=5)

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
