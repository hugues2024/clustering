# =========================
# IMPORT DES LIBRAIRIES
# =========================
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
df = pd.read_csv("cereals.csv")

# Colonnes ignorées comme dans WEKA
df_numeric = df.drop(columns=["name", "mfr", "type"])

# =========================
# 2. PRÉTRAITEMENT
# =========================

# ReplaceMissingValues → moyenne
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_numeric)

# Normalize → valeurs entre 0 et 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_scaled = pd.DataFrame(X_scaled, columns=df_numeric.columns)

# =========================
# 3. FONCTION K-MEANS
# =========================
def executer_kmeans(k):
    print(f"\n{'='*25} K-MEANS AVEC k = {k} {'='*25}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=100,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    df_res = X_scaled.copy()
    df_res["Cluster"] = labels

    # Moyennes par cluster
    moyennes = df_res.groupby("Cluster").mean().T
    print("\nMoyennes des attributs par cluster :")
    print(moyennes)

    # Répartition
    print("\nRépartition des instances :")
    counts = df_res["Cluster"].value_counts().sort_index()
    for c, n in counts.items():
        print(f"Cluster {c} : {n} instances ({n/len(df_res)*100:.1f}%)")

    # Qualité du clustering
    silhouette = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score (k={k}) : {silhouette:.4f}")

    return silhouette

# =========================
# 4. EXÉCUTION
# =========================
sil_k5 = executer_kmeans(5)
sil_k3 = executer_kmeans(3)

# =========================
# 5. COMPARAISON FINALE
# =========================
print("\n========== COMPARAISON FINALE ==========")
print(f"Silhouette k=5 : {sil_k5:.4f}")
print(f"Silhouette k=3 : {sil_k3:.4f}")

if sil_k5 > sil_k3:
    print("La solution k=5 est la plus satisfaisante.")
else:
    print("La solution k=3 est la plus satisfaisante.")
