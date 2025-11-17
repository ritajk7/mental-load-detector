import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Read data from Excel
data_path = os.path.join("data", "mental_load_data.xlsx")
df = pd.read_excel(data_path)

print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df.drop("mental_load_level", axis=1)
y = df["mental_load_level"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "mental_load_model.pkl")
joblib.dump(model, model_path)

print(f"\nModel saved to {model_path}")
