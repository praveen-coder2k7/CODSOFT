import pandas as pd

def preprocess(df):

    # Use correct Titanic dataset column names
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]].copy()

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df
