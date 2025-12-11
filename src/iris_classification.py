import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# set folder structure paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "Iris.csv")
OUTPUT_PATH = os.path.join(ROOT, "output_screenshots")
MODELS_PATH = os.path.join(ROOT, "models")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# load data
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")
print(df.head())

if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

# distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x="species", data=df)
plt.title("species Distribution")
plt.savefig(os.path.join(OUTPUT_PATH, "species_distribution.png"))
plt.close()

# pairplot
sns.pairplot(df, hue="species")
plt.savefig(os.path.join(OUTPUT_PATH, "pairplot.png"))
plt.close()

# prepare data
X = df.drop("species", axis=1)
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))
plt.close()

# save model
joblib.dump(model, os.path.join(MODELS_PATH, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

print("\nModel saved in models folder.")
print("All graphs saved in output_screenshots folder.")
