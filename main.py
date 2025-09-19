# Assignment 1 - SVM Classification on Breast Cancer Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

# --------------------------
# 1. Load Dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

print("Features:", data.feature_names[:5], "...")  # first few features
print("Classes:", data.target_names)

# --------------------------
# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------
# 3. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# 4. Train SVM Model
model = SVC(kernel='rbf', C=2.0, gamma='scale')
model.fit(X_train, y_train)

# --------------------------
# 5. Predictions
y_pred = model.predict(X_test)

# --------------------------
# 6. Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# --------------------------
# 7. Visualization (PCA for 2D projection)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Breast Cancer Dataset (PCA Reduced to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
