# Assignment 3: Dimensionality reduction using PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load data (30 features) and standardise
data = load_breast_cancer()
X, y = data.data, data.target
X_std = StandardScaler().fit_transform(X)
print("Original shape:", X.shape)

# 2. Fit PCA with all components -> explained variance
pca_full = PCA().fit(X_std)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = int(np.argmax(cum_var >= 0.95) + 1)
print("Variance explained by PC1, PC2:", pca_full.explained_variance_ratio_[:2].round(3))
print("Components needed for 95% variance:", n_95)

# 3. Reduce to 2 components for visualisation
X_2d = PCA(n_components=2).fit_transform(X_std)
print("Reduced shape (2D):", X_2d.shape)

# 4. Compare model accuracy: original vs PCA-reduced features
X_pca = PCA(n_components=n_95).fit_transform(X_std)
for name, feats in [("Original (30 features)", X_std), (f"PCA ({n_95} components)", X_pca)]:
    Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.25, random_state=42)
    acc = accuracy_score(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))
    print(f"Accuracy with {name}: {acc:.4f}")

# 5. Plots: scree/cumulative variance + 2D projection
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(range(1, 31), cum_var, "o-")
ax[0].axhline(0.95, color="r", ls="--"); ax[0].set_xlabel("Number of components")
ax[0].set_ylabel("Cumulative explained variance"); ax[0].set_title("Scree / Cumulative Variance")
sc = ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", s=15)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].set_title("2-D PCA Projection")
plt.tight_layout()
plt.savefig("a3_pca.png", dpi=130)
plt.show()
