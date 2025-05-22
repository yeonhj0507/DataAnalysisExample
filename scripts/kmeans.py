#!/usr/bin/env python
"""
K-Means clustering on Iris.
Elbow + Silhouette score → optimal k, cluster plot.
"""
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

iris = datasets.load_iris(as_frame=True)
X = StandardScaler().fit_transform(iris.data)
df = pd.DataFrame(X, columns=iris.feature_names)

# 1) Find optimal k (1–10)
inertias, silhouettes = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

opt_k = (silhouettes.index(max(silhouettes)) + 2)

# 2) Plot elbow & silhouette
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(range(2, 11), inertias, marker="o")
ax[0].set_title("Elbow (Inertia)")
ax[0].set_xlabel("k"), ax[0].set_ylabel("Inertia")

ax[1].plot(range(2, 11), silhouettes, marker="s")
ax[1].set_title("Silhouette")
ax[1].set_xlabel("k"), ax[1].set_ylabel("Score")

plt.suptitle(f"Optimal k ≈ {opt_k}")
plt.tight_layout()
plt.savefig(OUT / "kmeans_selection.png", dpi=150)
plt.close()

# 3) Fit final model
km_final = KMeans(n_clusters=opt_k, n_init="auto", random_state=42).fit(X)
df["cluster"] = km_final.labels_

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="cluster",
    palette="tab10",
    data=df,
)
plt.title(f"Iris K-Means – k={opt_k}")
plt.savefig(OUT / "kmeans_clusters.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ K-Means 완료 (k={opt_k}) – output/ 확인")
