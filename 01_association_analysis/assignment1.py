# Assignment 1: Association between dependent and independent variables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_diabetes

# 1. Load dataset (target = disease progression = dependent variable)
data = load_diabetes(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print(df.head(3).round(3))

# 2. Correlation matrix (Pearson)
corr = df.corr()
print("\nCorrelation of each independent variable with target:")
print(corr["target"].drop("target").sort_values(ascending=False).round(3))

# 3. Significance test (Pearson r and p-value) for each predictor
print("\nPearson r and p-value:")
for col in df.columns.drop("target"):
    r, p = stats.pearsonr(df[col], df["target"])
    print(f"{col:>4}: r = {r:6.3f}, p-value = {p:.4g}")

# 4. Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.tight_layout()
plt.savefig("a1_heatmap.png", dpi=130)
plt.show()

# 5. Scatter plot of the strongest predictor vs target
best = corr["target"].drop("target").abs().idxmax()
sns.regplot(x=df[best], y=df["target"], line_kws={"color": "red"})
plt.title(f"Target vs {best}")
plt.savefig("a1_scatter.png", dpi=130)
plt.show()
