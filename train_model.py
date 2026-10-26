import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "insurance_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "insurance_regression_model.pkl")

df = pd.read_csv(DATA_PATH)

TARGET = "charges"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove rows where the target is missing
df = df.dropna(subset=[TARGET]).copy()

# Separate features and target
X = df.drop(columns=[TARGET]).copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

# Remove rows with invalid target values
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(float)

# Remove columns that contain no useful values
empty_columns = X.columns[X.isna().all()].tolist()
if empty_columns:
    X = X.drop(columns=empty_columns)

feature_columns = X.columns.tolist()

# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    column for column in feature_columns
    if column not in numerical_features
]

print("Dataset loaded successfully!")
print("Rows:", len(X))
print("Target:", TARGET)
print("Features:", feature_columns)

# --------------------------------------------------
# 2. Prepare preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 3. Build and train the Random Forest model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining the Linear Regression model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate the model
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# --------------------------------------------------
# 5. Prepare information required by app.py
# --------------------------------------------------

category_options = {}

for column in categorical_features:
    values = X[column].dropna().unique().tolist()
    category_options[column] = sorted(
        [str(value) for value in values]
    )

numeric_defaults = {}

for column in numerical_features:
    median_value = X[column].median()
    numeric_defaults[column] = (
        float(median_value)
        if pd.notna(median_value)
        else 0.0
    )

model_bundle = {
    "model": model,
    "model_name": "Linear Regression",
    "feature_columns": feature_columns,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "category_options": category_options,
    "numeric_defaults": numeric_defaults,
    "target": TARGET,
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
}

# --------------------------------------------------
# 6. Save the new PKL model
# --------------------------------------------------

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model_bundle, file)

print("\nSUCCESS!")
print("New model saved to:")
print(MODEL_PATH)
print("\nYou can now run your Streamlit app.")