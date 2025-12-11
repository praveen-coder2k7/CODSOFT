import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import preprocess

# Load dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

# Preprocess
df_processed = preprocess(df.copy())

# Split
X = df_processed.drop("Survived", axis=1)
y = df_processed["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/final_model.pkl")

print("Model trained successfully!")
