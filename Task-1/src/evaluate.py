import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess

df = pd.read_csv("../data/Titanic-Dataset.csv")
df_processed = preprocess(df.copy())

X = df_processed.drop("Survived", axis=1)
y = df_processed["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("../models/final_model.pkl")

y_pred = model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===\n")
print(classification_report(y_test, y_pred))
