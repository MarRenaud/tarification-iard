# Moteur de tarification prédictif IARD

Projet de semestre MAM 4A — Polytech Lyon  
Encadrant : M. Soler

## Description
Modélisation de la prime d'assurance automobile à partir du jeu de données freMTPL2
(678 000 polices françaises) en utilisant des GLM (Poisson + Gamma).

## Équipe
- Marion — GLM Poisson + exploration des données
- Mario — GLM Gamma + prime pure + validation
- Cheikhna — Comparaison modèles + application Streamlit

## Installation

### 1. Cloner le repo
git clone https://forge.univ-lyon1.fr/p2110892/tarification-iard.git
cd tarification-iard

### 2. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Télécharger les données
Les données ne sont pas sur Git (trop lourdes). Lance ce script :

python3 -c "
import urllib.request
urllib.request.urlretrieve('https://www.openml.org/data/get_csv/20649148/freMTPL2freq.csv', 'data/freMTPL2freq.csv')
urllib.request.urlretrieve('https://www.openml.org/data/get_csv/20649149/freMTPL2sev.csv', 'data/freMTPL2sev.csv')
print('Données téléchargées !')
"

## Structure du projet
- data/        — données brutes (non versionnées)
- notebooks/   — exploration et analyses
- src/         — code Python réutilisable
- models/      — modèles exportés
- app/         — application Streamlit
- report/      — rapport final

## Pistes d'amélioration

### Axes statistiques
**Sur le GLM Poisson :**
- Tester la surdispersion — passer à Quasi-Poisson ou Binomial Négatif si nécessaire
- Ajouter des interactions entre variables (BonusMalus × DrivAge, Region × Density)
- Tester log(BonusMalus) pour mieux capturer l'effet non-linéaire

**Sur la modélisation globale :**
- Ajouter un modèle Tweedie et comparer avec l'approche Poisson × Gamma
- Entraîner un XGBoost et comparer avec le GLM (interprétabilité vs performance)
- Sélection de variables automatique par AIC/BIC

### Axes méthodologiques
- Segmentation du portefeuille en classes de risque homogènes
- Credibility theory (Bühlmann) pour pondérer les estimations individuelles
- Analyse de profitabilité : primes encaissées vs sinistres payés par segment

### Axes applicatifs
- Carte interactive plotly des primes moyennes par région
- Déploiement sur Streamlit Cloud avec URL publique
- Comparaison de deux profils côte à côte
- Graphique waterfall des contributions de chaque variable à la prime