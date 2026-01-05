import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# CONFIG STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Clustering Facebook Posts",
    layout="wide"
)

st.title("📊 Application de Clustering – Facebook Live Data")
st.write("Clustering des publications Facebook avec K-Means")

# --------------------------------------------------
# CHARGEMENT DES DONNÉES
# --------------------------------------------------
@st.cache_data
def load_data():
    return df = pd.read_csv("Live_20210128.csv")

df = load_data()

st.subheader("Aperçu du dataset")
st.dataframe(df.head())

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
df_clean = df.drop(
    columns=[
        "status_id",
        "status_published",
        "Column1",
        "Column2",
        "Column3",
        "Column4"
    ]
)

# Encodage de la variable catégorielle
df_clean = pd.get_dummies(df_clean, columns=["status_type"], drop_first=True)

# --------------------------------------------------
# NORMALISATION
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# --------------------------------------------------
# SIDEBAR - PARAMÈTRES
# --------------------------------------------------
st.sidebar.header("Paramètres")

k = st.sidebar.slider(
    "Nombre de clusters (k)",
    min_value=2,
    max_value=6,
    value=3
)

# --------------------------------------------------
# KMEANS + ÉVALUATION
# --------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clean["cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

# --------------------------------------------------
# PCA POUR VISUALISATION
# --------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean["PCA1"] = X_pca[:, 0]
df_clean["PCA2"] = X_pca[:, 1]

# --------------------------------------------------
# MENU DES FONCTIONNALITÉS
# --------------------------------------------------
st.subheader("Fonctionnalités de visualisation")

option = st.selectbox(
    "Choisissez une option",
    (
        "Voir les clusters (scatterplot)",
        "Voir toutes les classes",
        "Voir les classes séparément",
        "Voir les individus par classe"
    )
)

# --------------------------------------------------
# VISUALISATIONS
# --------------------------------------------------
if option == "Voir les clusters (scatterplot)":
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clean,
        x="PCA1",
        y="PCA2",
        hue="cluster",
        palette="Set1",
        ax=ax
    )
    ax.set_title("Clusters visualisés avec PCA")
    st.pyplot(fig)

elif option == "Voir toutes les classes":
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clean,
        x="PCA1",
        y="PCA2",
        hue="cluster",
        ax=ax
    )
    ax.set_title("Toutes les classes ensemble")
    st.pyplot(fig)

elif option == "Voir les classes séparément":
    selected_cluster = st.selectbox(
        "Choisir un cluster",
        sorted(df_clean["cluster"].unique())
    )

    subset = df_clean[df_clean["cluster"] == selected_cluster]

    fig, ax = plt.subplots()
    ax.scatter(subset["PCA1"], subset["PCA2"])
    ax.set_title(f"Cluster {selected_cluster}")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    st.pyplot(fig)

elif option == "Voir les individus par classe":
    selected_cluster = st.selectbox(
        "Choisir un cluster",
        sorted(df_clean["cluster"].unique())
    )

    st.write(f"Individus du cluster {selected_cluster}")
    st.dataframe(df_clean[df_clean["cluster"] == selected_cluster])

# --------------------------------------------------
# INTERPRÉTATION DES CLUSTERS
# --------------------------------------------------
st.subheader("Interprétation des clusters (moyennes)")
st.dataframe(df_clean.groupby("cluster").mean())


