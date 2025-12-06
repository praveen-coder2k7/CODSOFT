import pandas as pd
import joblib
from preprocess import preprocess

# Load model
model = joblib.load("models/final_model.pkl")

def predict_passenger(data):
    df = pd.DataFrame([data])

    # Add dummy Survived column for preprocessing
    df["Survived"] = 0

    df_processed = preprocess(df)
    df_processed = df_processed.drop("Survived", axis=1)

    pred = model.predict(df_processed)[0]
    return "Survived" if pred == 1 else "Not Survived"


# Test example
sample_passenger = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80.00
}


print("Prediction:", predict_passenger(sample_passenger))
