import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import joblib
import warnings
import requests

# =============================================================================
# 1. CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Data Mining : Insertion des Masters",
    page_icon="🚀",
    layout="wide",
)
warnings.filterwarnings("ignore", category=FutureWarning)

st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .stMetric { border-left: 5px solid #007bff; padding: 1rem;
                border-radius: 0.5rem; background-color: #262730; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; white-space: pre-wrap; background-color: #1a1a2e;
        border-radius: 6px 6px 0px 0px; border: 1px solid #3c3c5a;
        padding: 10px 18px; font-size: 15px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730; color: #ffffff;
        border-bottom: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. FONCTIONS DE CHARGEMENT AVEC CACHE
# =============================================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['qualite_emploi'] = pd.Categorical(df['qualite_emploi'], categories=['Standard', 'Élevée', 'Excellente'], ordered=True)
    df['parite'] = pd.Categorical(df['parite'], categories=['Majorité Masculine', 'Mixte', 'Majorité Féminine'], ordered=True)
    df['cluster'] = df['cluster'].astype(str)
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# =============================================================================
# 3. CHARGEMENT INITIAL DES DONNÉES ET RESSOURCES
# =============================================================================
try:
    df = load_data('insertion_pro_master_final_v2.csv')
    model_regression = load_model('salary_predictor_model.joblib')
    wordcloud_img1 = Image.open("wordcloud_article1.png")
    wordcloud_img2 = Image.open("wordcloud_article2.png")
    geojson_url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    geojson = requests.get(geojson_url).json()
except Exception as e:
    st.error(f"❌ Fichier manquant ou erreur de chargement : {e}. Veuillez exécuter le notebook `data_management.ipynb` et vérifier votre connexion.")
    st.stop()


# =============================================================================
# 4. BARRE LATÉRALE DE CONTRÔLE
# =============================================================================
with st.sidebar:
    st.title("🚀 Centre de Contrôle")
    st.write("Filtrez l'analyse pour explorer les données.")
    years = sorted(df['annee_diplome'].unique(), reverse=True)
    years_list = ['Toutes les années'] + years
    selected_year = st.selectbox("Année du diplôme", options=years_list)
    regions_list = ['France entière'] + sorted(df['region'].unique().tolist())
    selected_region = st.selectbox("Région", options=regions_list)

# =============================================================================
# 5. LOGIQUE DE FILTRAGE
# =============================================================================
if selected_year == 'Toutes les années':
    df_filtered = df.copy()
else:
    df_filtered = df[df['annee_diplome'] == selected_year]
if selected_region != 'France entière':
    df_filtered = df_filtered[df_filtered['region'] == selected_region]

# =============================================================================
# 6. INTERFACE PRINCIPALE
# =============================================================================
st.title("Projet de Data Mining : Analyse de l'Insertion des Masters")

tabs = st.tabs([
    "🏠 Accueil", "📊 Vue d'Ensemble", "📰 Analyse de Presse",
    "🗺️ Géographie", "👥 Analyse Sociale", "✨ Clustering",
    "🧠 Prédiction", "🗃️ Exploration"
])

# --- Onglet Accueil ---
with tabs[0]:
    st.header("Objectif du Projet et Méthodologie")
    st.markdown("Ce dashboard est une application de la méthodologie **CRISP-DM** visant à analyser en profondeur les débouchés des diplômés de Master en France. L'objectif est de transformer des données brutes en **insights stratégiques**.")
    st.subheader("Création de Variables Stratégiques")
    st.success("**1. Grand Domaine :** Agrégation intelligente des disciplines.")
    st.success("**2. Indice d'Attractivité :** Score combinant salaire (60%) et insertion (40%).")
    st.success("**3. Qualité de l'Emploi :** Indice qualitatif (stabilité, statut cadre).")
    st.success("**4. Parité :** Catégorisation de la part de femmes.")
    st.success("**5. Région administrative :** Mapping des académies vers les régions.")

# --- Onglet Vue d'Ensemble du Dataset ---
with tabs[1]:
    st.header("Présentation du Jeu de Données")
    st.markdown(f"Les données couvrent **{df['annee_diplome'].nunique()} années** (de {df['annee_diplome'].min()} à {df['annee_diplome'].max()}), et concernent **{df.shape[0]:,}** observations après nettoyage.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution par Année")
        st.bar_chart(df['annee_diplome'].value_counts().sort_index())
    with col2:
        st.subheader("Distribution par Domaine")
        domain_counts = df['grand_domaine'].value_counts()
        fig_pie = px.pie(domain_counts, values=domain_counts.values,
                         names=domain_counts.index, title="Répartition")
        st.plotly_chart(fig_pie, use_container_width=True)
    st.dataframe(df.head(10))

# --- Onglet Analyse de Presse ---
with tabs[2]:
    st.header("Analyse Comparative de la Presse (Text Mining)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vue Générale (Le Monde)")
        st.image(wordcloud_img1, use_container_width=True)
        with st.expander("Lire le résumé et voir la source"):
            st.markdown("**Résumé :** L'article souligne une conjoncture très positive en 2022, avec un taux d'emploi record pour les masters (93%) et des salaires en hausse, malgré l'inflation.\n\n**Source :** [Le Monde](https://www.lemonde.fr/campus/article/2023/12/13/l-insertion-professionnelle-des-jeunes-diplomes-a-atteint-un-niveau-record-en-2022_6205561_4401467.html)")
    with col2:
        st.subheader("Focus Social (APEC)")
        st.image(wordcloud_img2, use_container_width=True)
        with st.expander("Lire le résumé et voir la source"):
            st.markdown("**Résumé :** L'APEC nuance ce tableau en pointant des difficultés persistantes pour les jeunes femmes. Elles accèdent moins souvent au statut cadre et aux CDI, et un écart de salaire demeure.\n\n**Source :** [APEC](https://www.apec.fr/recruteur/marche-emploi/les-etudes-de-lapec/toutes-les-etudes/insertion-des-jeunes-diplomes-2024.html)")

# --- Onglet Analyse Géographique ---
with tabs[3]:
    st.header(f"Analyse Géographique ({selected_region}, {selected_year})")
    options = ['Tous les domaines'] + df_filtered['grand_domaine'].unique().tolist()
    sel_domain = st.selectbox("Affiner par Grand Domaine :", options=options)
    df_geo = df_filtered[df_filtered['grand_domaine'] == sel_domain] if sel_domain != 'Tous les domaines' else df_filtered
    st.subheader(f"Carte de l'Indice d'Attractivité ({sel_domain})")
    col_map, col_domtom = st.columns([3, 1])
    with col_map:
        st.markdown("**France Métropolitaine**")
        map_data = df_geo.groupby('region')['indice_attractivite'].mean().reset_index()
        fig_map = px.choropleth(map_data, geojson=geojson, featureidkey="properties.nom", locations="region", color="indice_attractivite", color_continuous_scale="Viridis", scope="europe", labels={'indice_attractivite': "Indice"})
        fig_map.update_geos(fitbounds=False, visible=False, center={"lat": 46.8, "lon": 2.35}, projection_scale=4.5)
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    with col_domtom:
        st.markdown("**Outre-mer**")
        domtom_list = ['Guadeloupe', 'Guyane', 'Martinique', 'Mayotte', 'La Réunion']
        df_domtom = df_filtered[df_filtered['region'].isin(domtom_list)]
        if not df_domtom.empty:
            domtom_data = df_domtom.groupby('region')['indice_attractivite'].mean().sort_values().reset_index()
            fig_domtom = px.bar(domtom_data, y='region', x='indice_attractivite', orientation='h', color='indice_attractivite', color_continuous_scale="Viridis", labels={'region': '', 'indice_attractivite': "Indice"})
            fig_domtom.update_layout(height=500, yaxis_title=None)
            st.plotly_chart(fig_domtom, use_container_width=True)

# --- Onglet Analyse Sociale ---
with tabs[4]:
    st.header(f"Analyses Sectorielles et Sociales ({selected_region}, {selected_year})")
    st.subheader("Positionnement Stratégique des Grands Domaines")
    st.info("**Comment lire ce graphique ?** Chaque bulle est un domaine. L'idéal est en haut à droite (haut salaire, haute insertion).")
    domain_agg = df_filtered.groupby('grand_domaine').agg(salaire_moyen=('salaire_median', 'mean'), insertion_moyenne=('taux_insertion', 'mean'), effectif=('discipline', 'count')).reset_index()
    fig_bubble = px.scatter(domain_agg, x="insertion_moyenne", y="salaire_moyen", size="effectif", color="grand_domaine", hover_name="grand_domaine", size_max=60, labels={"insertion_moyenne": "Taux d'Insertion Moyen (%)", "salaire_moyen": "Salaire Médian Moyen (€)"})
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.markdown("---")
    st.subheader("Analyse de la Qualité de l'Emploi par Domaine")
    st.info("**Comment lire ce graphique ?** Chaque barre représente 100% d'un domaine. On y voit la part des emplois de qualité 'Standard', 'Élevée' ou 'Excellente'.")
    quality_data = df_filtered.groupby(['grand_domaine', 'qualite_emploi']).size().unstack(fill_value=0)
    quality_data_percent = quality_data.div(quality_data.sum(axis=1), axis=0)
    fig_quality = px.bar(quality_data_percent, x=quality_data_percent.index, y=quality_data_percent.columns, labels={"x": "Grand Domaine", "value": "Pourcentage", "variable": "Qualité de l'Emploi"}, color_discrete_map={"Standard": "#636EFA", "Élevée": "#00CC96", "Excellente": "#FFA15A"})
    fig_quality.update_layout(yaxis_tickformat=".0%", yaxis_title="Répartition en %", barmode='stack', legend_title="Qualité")
    st.plotly_chart(fig_quality, use_container_width=True)
    st.markdown("---")
    st.subheader("Salaires et Parité par Domaine")
    st.info("**Comment lire ce graphique ?** Chaque boîte montre la distribution des salaires pour une catégorie. Cela permet de voir si, au sein d'un même domaine, les filières majoritairement féminines, mixtes ou masculines ont des perspectives salariales différentes.")
    df_gender = df_filtered.dropna(subset=['parite'])
    if not df_gender.empty:
        fig_box = px.box(df_gender, x="grand_domaine", y="salaire_median", color="parite", category_orders={"parite": ['Majorité Masculine', 'Mixte', 'Majorité Féminine']}, labels={"grand_domaine": "", "salaire_median": "Distribution des Salaires (€)", "parite": "Parité"})
        st.plotly_chart(fig_box, use_container_width=True)

# --- Onglet Clustering ---
with tabs[5]:
    st.header("Data Mining Descriptif : Segmentation par Clustering K-Means")
    st.markdown("Nous avons utilisé **K-Means** pour regrouper les académies en 4 profils distincts sur la base de leurs performances globales.")
    
    cluster_names = {'0': "Pôles d'Excellence", '1': "Régions en Développement", '2': "Académies Spécifiques", '3': "Pôles Dynamiques"}
    cluster_colors = {"Pôles d'Excellence": "#EF553B", "Régions en Développement": "#636EFA", "Académies Spécifiques": "#00CC96", "Pôles Dynamiques": "#AB63FA"}

    st.subheader("1. La Géographie des Clusters")
    st.info("Cette carte colore chaque région selon le profil dominant de ses académies. Elle révèle des ensembles géographiques cohérents.")
    
    region_cluster = df.groupby('region')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    region_cluster['Nom du Cluster'] = region_cluster['cluster'].map(cluster_names)
    
    fig_map_cluster = px.choropleth(
        region_cluster, geojson=geojson, featureidkey="properties.nom",
        locations="region", color="Nom du Cluster",
        color_discrete_map=cluster_colors,
        scope="europe", labels={'Nom du Cluster': "Profil"}
    )
    fig_map_cluster.update_geos(fitbounds=False, visible=False, center={"lat": 46.8, "lon": 2.35}, projection_scale=4.5)
    fig_map_cluster.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
    st.plotly_chart(fig_map_cluster, use_container_width=True)

    with st.expander("Cliquez ici pour comprendre la signification de chaque profil"):
        st.markdown("""
        - **<span style='color:#EF553B;'>Pôles d'Excellence</span> :** Salaires les plus bas mais stabilité d'emploi correcte.
        - **<span style='color:#636EFA;'>Régions en Développement</span> :** Salaires et taux d'insertion légèrement inférieurs à la moyenne.
        - **<span style='color:#00CC96;'>Académies Spécifiques</span> :** Salaires les plus élevés (typiquement l'Île-de-France).
        - **<span style='color:#AB63FA;'>Pôles Dynamiques</span> :** Le meilleur équilibre avec les meilleurs taux d'insertion et de stabilité.
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. L'ADN de chaque Cluster")
    st.info("Pour une lecture claire, les métriques sont présentées sur deux graphiques avec des échelles adaptées.")
    
    cluster_centers = df.groupby('cluster').agg(
        salaire_median=('salaire_median', 'mean'),
        taux_insertion=('taux_insertion', 'mean'),
        taux_emploi_stable=('taux_emploi_stable', 'mean')
    ).reset_index()
    cluster_centers['Nom du Cluster'] = cluster_centers['cluster'].map(cluster_names)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_sal = px.bar(
            cluster_centers, x='Nom du Cluster', y='salaire_median',
            title="Comparaison des Salaires Médians",
            labels={'Nom du Cluster': '', 'salaire_median': 'Salaire Médian Moyen (€)'},
            color='Nom du Cluster', color_discrete_map=cluster_colors
        )
        fig_sal.update_layout(showlegend=False)
        st.plotly_chart(fig_sal, use_container_width=True)
    with col2:
        melted_rates = cluster_centers.melt(
            id_vars='Nom du Cluster', value_vars=['taux_insertion', 'taux_emploi_stable']
        )
        fig_rates = px.bar(
            melted_rates, x='Nom du Cluster', y='value', color='variable',
            barmode='group', title="Comparaison des Taux d'Insertion et de Stabilité",
            labels={'Nom du Cluster': '', 'value': 'Taux Moyen (%)', 'variable': 'Métrique'},
            color_discrete_map={'taux_insertion': '#1f77b4', 'taux_emploi_stable': '#ff7f0e'}
        )
        fig_rates.update_layout(yaxis_range=[0,100])
        st.plotly_chart(fig_rates, use_container_width=True)

# --- Onglet Prédiction ---
with tabs[6]:
    st.header("Data Mining Prédictif : Modélisation par Régression")
    st.subheader("Outil de Prédiction de Salaire")
    st.markdown(
        "Ce modèle **Random Forest** a été entraîné pour estimer le salaire "
        "médian d'un diplômé en fonction de son grand domaine et de sa région."
    )
    col1, col2 = st.columns(2)
    with col1:
        pred_region = st.selectbox("Choisissez une Région",
                                   options=sorted(df['region'].unique()))
    with col2:
        pred_domain = st.selectbox("Choisissez un Grand Domaine",
                                   options=sorted(df['grand_domaine'].unique()))
    if st.button("Estimer le salaire"):
        input_data = pd.DataFrame(
            {'grand_domaine': [pred_domain], 'region': [pred_region]}
        )
        predicted_salary = model_regression.predict(input_data)[0]
        st.metric(label="Salaire Médian Mensuel Estimé",
                  value=f"{int(predicted_salary)} €")

# --- Onglet Exploration ---
with tabs[7]:
    st.header("Exploration des Données Finales")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)