# Assignment 7: Support Vector Machine for Regression (SVR)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate non-linear data
rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(200, 1), axis=0)
y = np.sin(X).ravel() + rng.normal(0, 0.15, X.shape[0])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Compare kernels
models = {
    "Linear": SVR(kernel="linear", C=1.0),
    "Polynomial": SVR(kernel="poly", degree=3, C=1.0),
    "RBF": SVR(kernel="rbf", C=10, gamma=0.5, epsilon=0.1),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name:10s} R2 = {r2_score(yte, p):.4f} | RMSE = {np.sqrt(mean_squared_error(yte, p)):.4f}")

# 3. Hyper-parameter tuning with GridSearchCV (RBF kernel)
pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
grid = {"svr__C": [0.1, 1, 10, 100],
        "svr__gamma": [0.01, 0.1, 1],
        "svr__epsilon": [0.05, 0.1, 0.2]}
gs = GridSearchCV(pipe, grid, cv=5, scoring="r2").fit(Xtr, ytr)
print("\nBest params:", gs.best_params_)
print("Best CV R2 :", round(gs.best_score_, 4))
print("Test  R2   :", round(r2_score(yte, gs.predict(Xte)), 4))

# 4. Visualise fit
Xgrid = np.linspace(0, 5, 300).reshape(-1, 1)
plt.scatter(X, y, s=15, color="grey", label="Data")
plt.plot(Xgrid, models["RBF"].predict(Xgrid), "r", lw=2, label="SVR (RBF)")
plt.plot(Xgrid, models["Linear"].predict(Xgrid), "g--", label="SVR (Linear)")
sv = models["RBF"].support_
plt.scatter(Xtr[sv], ytr[sv], s=40, facecolors="none", edgecolors="b", label="Support vectors")
plt.legend(); plt.title("Support Vector Regression")
plt.savefig("a7_svr.png", dpi=130); plt.show()
