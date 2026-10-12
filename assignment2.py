# Assignment 2: Clustering techniques using unsupervised learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# 1. Create / load data
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Elbow method and silhouette score to choose K
inertia, sil = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={sil[-1]:.3f}")

best_k = list(K_range)[int(np.argmax(sil))]
print("Best K by silhouette score:", best_k)

# 3. Apply the three clustering algorithms
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X)

print("\nSilhouette  K-Means      :", round(silhouette_score(X, kmeans.labels_), 3))
print("Silhouette  Hierarchical :", round(silhouette_score(X, hier.labels_), 3))
mask = dbscan.labels_ != -1
print("DBSCAN clusters found    :", len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0))
print("DBSCAN noise points      :", int((~mask).sum()))

# 4. Visualise
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].plot(list(K_range), inertia, "o-"); ax[0].set_title("Elbow Method")
ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
for a, labels, t in zip(ax[1:], [kmeans.labels_, hier.labels_, dbscan.labels_],
                        ["K-Means", "Hierarchical", "DBSCAN"]):
    a.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
    a.set_title(t)
plt.tight_layout()
plt.savefig("a2_clusters.png", dpi=130)
plt.show()
