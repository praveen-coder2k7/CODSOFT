import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------
print("Loading dataset...")

df = pd.read_csv("Data/IMDb Movies India.csv", encoding="latin-1")

print("Dataset loaded successfully!")
print(df.head())

# ------------------------------------
# STEP 2: BASIC INFO
# ------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------
# STEP 3: DATA CLEANING
# ------------------------------------
print("\nCleaning data...")


df = df.dropna(subset=['Rating'])


df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')


df['Duration'] = df['Duration'].astype(str).str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')


df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')


df['Year'].fillna(df['Year'].median(), inplace=True)
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Votes'].fillna(df['Votes'].median(), inplace=True)

df['Genre'].fillna('Unknown', inplace=True)
df['Director'].fillna('Unknown', inplace=True)
df['Actor 1'].fillna('Unknown', inplace=True)

print("Data cleaning completed!")
print(df.isnull().sum())

# ------------------------------------
# STEP 4: FEATURE ENGINEERING
# ------------------------------------
print("\nFeature engineering...")

df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])
# ------------------------------------
# STEP 5: ENCODING CATEGORICAL DATA
# ------------------------------------
print("Encoding categorical columns...")

label_encoders = {}

categorical_cols = ['Genre', 'Director', 'Actor 1']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Encoding completed!")

# ------------------------------------
# STEP 6: FEATURE SELECTION
# ------------------------------------
print("\nSelecting features and target...")

X = df[['Genre', 'Director', 'Actor 1', 'Year', 'Duration', 'Votes']]
y = df['Rating']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------------
# STEP 7: TRAIN-TEST SPLIT
# ------------------------------------
print("\nSplitting data into train and test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# ------------------------------------
# STEP 8: TRAIN MODELS
# ------------------------------------
print("\nTraining Linear Regression model...")

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Training Random Forest model...")

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ------------------------------------
# STEP 9: MODEL EVALUATION
# ------------------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score:", r2_score(y_true, y_pred))

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ------------------------------------
# STEP 10: NEW MOVIE PREDICTION
# ------------------------------------
print("\nPredicting rating for a new movie...")

new_movie = pd.DataFrame({
    'Genre': [label_encoders['Genre'].transform(['Drama'])[0]],
    'Director': [label_encoders['Director'].transform(['Rajkumar Hirani'])[0]],
    'Actor 1': [label_encoders['Actor 1'].transform(['Aamir Khan'])[0]],
    'Year': [2023],
    'Duration': [160],
    'Votes': [100000]
})

predicted_rating = rf.predict(new_movie)
print("Predicted Movie Rating:", round(predicted_rating[0], 2))
