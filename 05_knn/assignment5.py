# Assignment 5: K-Nearest Neighbours classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data and split
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)

# 2. Find best K using 5-fold cross validation (scaling is essential for KNN)
k_values = range(1, 26)
cv_scores = []
for k in k_values:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv_scores.append(cross_val_score(pipe, X_train, y_train, cv=5).mean())
best_k = list(k_values)[int(np.argmax(cv_scores))]
print("Best K:", best_k, "| CV accuracy:", round(max(cv_scores), 4))

# 3. Train final model
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 4. Evaluate
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Accuracy vs K plot
plt.plot(list(k_values), cv_scores, "o-")
plt.axvline(best_k, color="r", ls="--", label=f"Best K = {best_k}")
plt.xlabel("K"); plt.ylabel("CV Accuracy"); plt.title("KNN - Choosing K")
plt.legend(); plt.savefig("a5_knn.png", dpi=130); plt.show()
