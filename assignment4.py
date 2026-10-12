# Assignment 4: Naive Bayes classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# 1. Load data and split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

# 2. Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 3. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("5-fold CV accuracy:", cross_val_score(model, iris.data, iris.target, cv=5).mean().round(4))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. Predict probability for a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Predicted class :", iris.target_names[model.predict(sample)[0]])
print("Class probabilities:", model.predict_proba(sample).round(4))

# 5. Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Naive Bayes - Confusion Matrix")
plt.savefig("a4_cm.png", dpi=130)
plt.show()

# --- Bonus: Multinomial NB for text classification ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = ["win money now", "free lottery offer", "meeting at 10 am",
         "project report attached", "claim your free prize", "lunch tomorrow?"]
labels = [1, 1, 0, 0, 1, 0]            # 1 = spam, 0 = not spam
cv = CountVectorizer()
nb = MultinomialNB().fit(cv.fit_transform(texts), labels)
print("\n'free money offer' ->", "Spam" if nb.predict(cv.transform(["free money offer"]))[0] else "Not spam")
