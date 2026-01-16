# IMPORT DES LIBRAIRIES
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1 CHARGEMENT DES DONNÉES
df = pd.read_csv("cereals.csv")

# Colonnes ignorées comme dans WEKA
df_numeric = df.drop(columns=["name", "mfr", "type"])

# 2 PRÉTRAITEMENT (IDENTIQUE WEKA)

# ReplaceMissingValues → moyenne
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_numeric)

# Normalize → [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_scaled = pd.DataFrame(X_scaled, columns=df_numeric.columns)

# 3 FONCTION K-MEANS
def executer_kmeans(k):
    print(f"\n{'='*25} K-MEANS AVEC k = {k} {'='*25}")

    kmeans = KMeans(
        n_clusters=k,
        init="random",        # initialisation aléatoire comme WEKA
        max_iter=500,         # -I 500
        n_init=10,            # redémarrages multiples
        random_state=10       # -S 10
    )

    labels = kmeans.fit_predict(X_scaled)

    df_res = X_scaled.copy()
    df_res["Cluster"] = labels

    # Moyennes par cluster (centroïdes finaux)
    moyennes = df_res.groupby("Cluster").mean().T
    print("\nMoyennes des attributs par cluster :")
    print(moyennes)

    # Répartition des instances
    print("\nRépartition des instances :")
    counts = df_res["Cluster"].value_counts().sort_index()
    for c, n in counts.items():
        print(f"Cluster {c} : {n} instances ({n/len(df_res)*100} %)")

    # SSE (équivalent WEKA)
    print("\nWithin-cluster sum of squared errors (SSE) :")
    print(kmeans.inertia_)

    # Silhouette (qualité globale)
    silhouette = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score (k={k}) : {silhouette}")

    return silhouette, kmeans.inertia_

# 4 EXÉCUTION
sil_k5, sse_k5 = executer_kmeans(5)
sil_k3, sse_k3 = executer_kmeans(3)
 
# 5 COMPARAISON FINALE
print("\n========== COMPARAISON FINALE ==========")
print(f"k=5 → Silhouette : {sil_k5} | SSE : {sse_k5}")
print(f"k=3 → Silhouette : {sil_k3} | SSE : {sse_k3}")

if sil_k5 > sil_k3:
    print("La solution k=5 est la plus satisfaisante.")
else:
    print("La solution k=3 est la plus satisfaisante.")