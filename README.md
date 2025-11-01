# projets-machine-learning
# XGBoost Credit Default Prediction

## Description
Ce projet utilise **XGBoostClassifier** pour prédire si un client va rembourser son crédit ou non, à partir du dataset **UCI Credit Default Dataset**.  
Le projet inclut la préparation des données, le feature engineering et l’optimisation des hyperparamètres pour améliorer la performance du modèle.

## Objectif
Prédire le risque de défaut de paiement d’un client.

## Dataset
[UCI Credit Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Étapes réalisées
- Feature engineering : âge, montant du crédit, historique de paiement, etc.  
- Gestion des valeurs manquantes et nettoyage des données  
- Optimisation des hyperparamètres : `max_depth`, `learning_rate`, `n_estimators`  
- Analyse de l’importance des variables  
- Évaluation des performances avec **AUC** et **F1-score**

## Technologies utilisées
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn

## Résultats

# SVM Handwritten Digit Classification

## Description
Ce projet utilise **SVC (Support Vector Classifier)** pour classer des images de chiffres manuscrits (0 à 9) à partir du dataset MNIST ou `sklearn.datasets.load_digits`.  
Le projet inclut la réduction de dimension avec **PCA**, la comparaison des kernels et l’analyse de la scalabilité.

## Objectif
Classer correctement les chiffres manuscrits en minimisant les erreurs.

## Dataset
- MNIST (ou `sklearn.datasets.load_digits`)

## Étapes réalisées
- Réduction de dimension avec PCA  
- Comparaison des kernels : linéaire, RBF, polynôme  
- Test de la scalabilité (temps d’entraînement)  
- Visualisation des marges et des erreurs de classification

## Technologies utilisées
- Python
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

## Résultats
- Comparaison des performances selon le kernel  
- Visualisation des marges et erreurs pour mieux comprendre le modèle  
- Précision finale sur les chiffres manuscrits optimisée grâce à PCA et choix du kernel.

# KMeans Customer Segmentation

## Description
Ce projet utilise **KMeans** pour segmenter les clients d’un site e-commerce en différents profils de comportement.  
L’objectif est d’identifier des groupes homogènes afin de proposer des actions marketing adaptées.

## Objectif
Segmenter les clients pour mieux comprendre leurs comportements et optimiser les stratégies commerciales.

## Dataset
[Online Retail Dataset (UCI)](https://archive.ics.uci.edu/dataset/352/online+retail)

## Étapes réalisées
- Sélection du nombre optimal de clusters via **méthode du coude** et **silhouette**  
- Analyse des clusters : profil client, panier moyen, fréquence d’achat  
- Réduction de dimension (PCA ou t-SNE) pour visualiser les clusters  
- Interprétation des résultats et recommandations marketing

## Technologies utilisées
- Python
- Pandas, NumPy
- Scikit-learn (KMeans, PCA, t-SNE)
- Matplotlib, Seaborn

## Résultats
- Visualisation des clusters pour compréhension intuitive  
- Analyse des comportements clients et recommandations pour chaque segment  
- Outil pratique pour guider les décisions marketing basées sur les données.

- Modèle capable de prédire les défauts avec une performance mesurée via **AUC** et **F1-score**.  
- Visualisation de l’importance des features pour interprétation métier.
