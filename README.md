# 🚀 Projet de Data Mining : Analyse de l'Insertion des Diplômés de Master

**Projet réalisé dans le cadre du module "Data Management, Data Visualisation & Text Mining" Sorbonne.**

Ce projet vise à analyser en profondeur les dynamiques d'insertion professionnelle des titulaires de Master en France, en transformant des données brutes en un outil d'aide à la décision interactif.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](LIEN_VERS_VOTRE_APP_STREAMLIT)  <!-- Mettez ici le lien si vous déployez l'app -->

## 🎯 Objectifs du Projet

L'objectif principal était de développer une application web interactive avec Streamlit pour :
- **Explorer** les données sur les taux d'emploi et les salaires.
- **Visualiser** les disparités géographiques et sectorielles.
- **Modéliser** les données pour découvrir des profils cachés (clustering) et prédire des tendances (régression).
- **Communiquer** les résultats de manière claire et actionnable.

## ⚙️ Méthodologie : CRISP-DM

Notre travail a rigoureusement suivi la méthodologie **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, la norme industrielle pour les projets de science des données.

1.  **Business Understanding :** Comprendre le besoin : aider les étudiants à s'orienter et les universités à se positionner.
2.  **Data Understanding :** Explorer le jeu de données de data.gouv.fr pour en comprendre la structure, les forces et les faiblesses.
3.  **Data Preparation :** Phase la plus critique.
    - **Nettoyage :** Traitement des valeurs manquantes (`'ns'`) et imputation intelligente par la médiane de groupe pour préserver la cohérence des données.
    - **Feature Engineering :** Création de 5 variables à forte valeur ajoutée (`Grand Domaine`, `Indice d'Attractivité`, `Qualité de l'Emploi`, `Parité`, `Région`) pour enrichir l'analyse.
4.  **Modeling :** Application de deux techniques de Data Mining.
    - **Clustering (K-Means) :** Pour segmenter les académies en 4 profils de performance distincts et découvrir des structures géographiques et économiques sous-jacentes.
    - **Régression (Random Forest) :** Pour créer un modèle prédictif capable d'estimer un salaire médian en fonction de la filière et de la région.
5.  **Evaluation :**
    - Analyse de la méthode du coude pour choisir le nombre optimal de clusters.
    - Analyse qualitative des résultats des modèles pour s'assurer qu'ils ont un "sens métier".
6.  **Deployment :**
    - Développement d'un dashboard interactif avec **Streamlit**.
    - Création de visualisations claires et commentées (cartes, boxplots, bubble charts, etc.) pour communiquer les résultats.
    - Intégration d'une analyse **Text Mining** comparative de la presse pour contextualiser les données chiffrées.

## 🔧 Environnement Technique

- **Langage :** Python 3
- **Librairies principales :**
  - **Analyse de données :** Pandas, NumPy
  - **Modélisation :** Scikit-learn
  - **Dataviz :** Plotly Express, Matplotlib
  - **Application Web :** Streamlit
  - **Text Mining :** WordCloud, NLTK

## 🚀 Comment Lancer le Projet

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/UtmostMaker/projet-datamining_insertion_master]
    cd projet-datamining_insertion_master
    ```
2.  **Créer un environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Télécharger les données NLTK :**
    ```python
    # Lancer python3 et exécuter :
    import nltk
    nltk.download('stopwords')
    ```
5.  **Exécuter l'application Streamlit :**
    ```bash
    streamlit run app.py
    ```

## 👥 Auteurs

- Prénom NOM 1
- Prénom NOM 2
- Prénom NOM 3
