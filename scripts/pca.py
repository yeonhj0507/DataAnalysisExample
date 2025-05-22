#!/usr/bin/env python
"""
PCA demo on the Iris dataset.
Saves 2-D scatter plot and explained-variance report.
"""
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

# 1) Load data
iris = datasets.load_iris(as_frame=True)
X, y = iris.data, iris.target
target_names = iris.target_names

# 2) Standardize
X_std = StandardScaler().fit_transform(X)

# 3) PCA → 2 components
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_std)
df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
df_pca["target"] = y

# 4) Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="target",
    palette="tab10",
    data=df_pca,
)
plt.title("Iris PCA – first two principal components")
plt.savefig(OUT / "iris_pca.png", dpi=150, bbox_inches="tight")
plt.close()

# 5) Variance ratio
var = pd.Series(
    pca.explained_variance_ratio_,
    index=["PC1", "PC2"],
    name="Explained_Variance"
)
var.to_csv(OUT / "iris_pca_variance.csv", index_label="Component")

print("✅ PCA 완료 – 결과는 output/ 폴더에서 확인하세요.")
