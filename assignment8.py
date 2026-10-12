# Assignment 8: Regularization to avoid overfitting (Ridge, Lasso, ElasticNet)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Small noisy dataset -> easy to overfit with a high-degree polynomial
rng = np.random.RandomState(0)
X = np.sort(rng.uniform(-3, 3, 40)).reshape(-1, 1)
y = 0.5 * X.ravel() ** 2 - X.ravel() + rng.normal(0, 1.0, 40)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1)

# 2. Fit models with degree-15 polynomial features
def build(reg):
    return make_pipeline(PolynomialFeatures(15, include_bias=False), StandardScaler(), reg)

models = {
    "No regularisation": build(LinearRegression()),
    "Ridge (L2)": build(Ridge(alpha=10)),
    "Lasso (L1)": build(Lasso(alpha=0.1, max_iter=100000)),
    "ElasticNet": build(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)),
}

print(f"{'Model':20s} {'Train MSE':>10s} {'Test MSE':>10s}")
for name, m in models.items():
    m.fit(Xtr, ytr)
    tr = mean_squared_error(ytr, m.predict(Xtr))
    te = mean_squared_error(yte, m.predict(Xte))
    print(f"{name:20s} {tr:10.3f} {te:10.3f}")

# 3. Number of coefficients shrunk to zero by Lasso
lasso_coef = models["Lasso (L1)"].named_steps["lasso"].coef_
print("\nLasso: non-zero coefficients =", int((lasso_coef != 0).sum()), "of", len(lasso_coef))

# 4. Plot fitted curves
Xg = np.linspace(-3, 3, 300).reshape(-1, 1)
plt.figure(figsize=(8, 5))
plt.scatter(Xtr, ytr, c="k", label="Train"); plt.scatter(Xte, yte, c="orange", label="Test")
for name, m in models.items():
    plt.plot(Xg, m.predict(Xg), label=name)
plt.ylim(-4, 12); plt.legend(); plt.title("Effect of Regularisation (degree 15)")
plt.savefig("a8_reg.png", dpi=130); plt.show()
