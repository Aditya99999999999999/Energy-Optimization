# =========================
# TITANIC ML PROJECT
# Preprocessing + Logistic Regression + KMeans + DBSCAN + ROC
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# =========================
# STEP 1: LOAD DATA
# =========================
data = pd.read_csv("titanic.csv")

# Basic cleanup (drop rows where 'Survived' is missing)
data = data.dropna(subset=['Survived'])

# Select key columns
X = data[['Age', 'Fare', 'Sex', 'Embarked']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# STEP 2: DEFINE FEATURES
# =========================
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked']

# =========================
# STEP 3: DEFINE PIPELINES
# =========================

# Numeric pipeline
numeric_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# =========================
# STEP 4: COLUMN TRANSFORMER
# =========================
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transform, numeric_features),
    ('cat', categorical_transform, categorical_features)
])

# =========================
# STEP 5: LOGISTIC REGRESSION PIPELINE (Supervised)
# =========================
clf = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
clf.fit(X_train, y_train)

# Evaluate accuracy
print("Logistic Regression Accuracy:", clf.score(X_test, y_test))

# =========================
# STEP 6: ROC CURVE
# =========================
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# =========================
# STEP 7: KMEANS CLUSTERING (Unsupervised)
# =========================
# Preprocess X (no labels)
X_scaled = preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

data['KMeans_Cluster'] = kmeans_labels

print("\nKMeans Cluster Counts:")
print(data['KMeans_Cluster'].value_counts())

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title("KMeans Clustering (2 Clusters)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.show()

# =========================
# STEP 8: DBSCAN CLUSTERING (Unsupervised)
# =========================
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

data['DBSCAN_Cluster'] = dbscan_labels

print("\nDBSCAN Cluster Counts:")
print(data['DBSCAN_Cluster'].value_counts())

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='plasma', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.show()
    