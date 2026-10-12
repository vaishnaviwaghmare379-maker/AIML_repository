# Assignment 6: Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Part A: Simple Linear Regression (BMI -> disease progression) ----------
data = load_diabetes(as_frame=True)
df = data.frame
X = df[["bmi"]]; y = df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

slr = LinearRegression().fit(Xtr, ytr)
pred = slr.predict(Xte)
print("Simple LR  -> slope: %.2f, intercept: %.2f" % (slr.coef_[0], slr.intercept_))
print("R2: %.4f | RMSE: %.2f" % (r2_score(yte, pred), np.sqrt(mean_squared_error(yte, pred))))

plt.scatter(Xte, yte, label="Actual")
plt.plot(Xte.sort_values("bmi"), slr.predict(Xte.sort_values("bmi")), "r", label="Regression line")
plt.xlabel("BMI (standardised)"); plt.ylabel("Disease progression")
plt.legend(); plt.title("Simple Linear Regression")
plt.savefig("a6_simple.png", dpi=130); plt.show()

# ---------- Part B: Multiple Linear Regression (all features) ----------
X = df.drop(columns="target")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
mlr = LinearRegression().fit(Xtr, ytr)
pred = mlr.predict(Xte)

print("\nMultiple LR coefficients:")
print(pd.Series(mlr.coef_, index=X.columns).round(2).to_string())
print("Intercept: %.2f" % mlr.intercept_)
print("MAE : %.2f" % mean_absolute_error(yte, pred))
print("RMSE: %.2f" % np.sqrt(mean_squared_error(yte, pred)))
print("R2  : %.4f" % r2_score(yte, pred))

# Actual vs Predicted
plt.scatter(yte, pred)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Multiple LR: Actual vs Predicted")
plt.savefig("a6_multiple.png", dpi=130); plt.show()

# ---------- Part C: From-scratch (Normal Equation) ----------
Xb = np.c_[np.ones(len(Xtr)), Xtr.values]
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ ytr.values
print("\nNormal-equation intercept: %.2f (matches sklearn: %s)" %
      (theta[0], np.isclose(theta[0], mlr.intercept_)))
