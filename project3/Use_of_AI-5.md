Use of AI
---------

[1]. Tool: ChatGPT
     Prompt: Write the import/setup cell, including reproducibility settings and an RMSE helper function.
     Output:
```python
# [Use of AI: 1] Imports, settings, and helper utilities.

import io
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
```

[2]. Tool: ChatGPT
     Prompt: Write Colab code to load the CarDekho CSV using file upload, with a non-Colab fallback to reading from a local path. Also include an optional Kaggle API download cell.
     Output:
```python
# [Use of AI: 2] Load the dataset by uploading CSV directly.

try:
    from google.colab import files  # type: ignore
    print("Colab detected. Please upload your CSV now...")
    uploaded = files.upload()  # choose the CarDekho CSV file
    filename = next(iter(uploaded))
    print(f"Loaded file: {filename}")
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
except Exception as e:
    print("Colab upload not available (or upload failed).")
    print("Fallback: place the CSV in the same folder and set CSV_PATH below.")
    CSV_PATH = "CAR DETAILS FROM CAR DEKHO.csv"  # <-- update if needed
    df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
df.head()


# Optional Kaggle cell
# [Use of AI: 2] OPTIONAL Kaggle API download (only if you want it).
# Steps:
# 1) In Kaggle: Account -> Create New API Token -> downloads kaggle.json
# 2) Upload kaggle.json when prompted below
# 3) Fill in DATASET_SLUG with the dataset identifier from Kaggle (e.g., "nehalbirla/vehicle-dataset-from-cardekho")

# !pip -q install kaggle
# from google.colab import files
# files.upload()  # upload kaggle.json

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# DATASET_SLUG = "nehalbirla/vehicle-dataset-from-cardekho"  # example
# !kaggle datasets download -d $DATASET_SLUG -p data --unzip

# df = pd.read_csv("data/CAR DETAILS FROM CAR DEKHO.csv")
# print(df.shape)
# df.head()
```

[3]. Tool: ChatGPT
     Prompt: Write a quick EDA cell that prints columns, missing values, shows basic describe output, and plots simple histograms for key numeric columns.
     Output:
```python
# [Use of AI: 3] Basic inspection + quick plots.

print("Columns:", list(df.columns))
display(df.describe(include="all").transpose().head(20))

print("\nMissing values per column:")
display(df.isna().sum().sort_values(ascending=False))

# Basic numeric histograms (only for columns that look numeric)
num_candidates = ["selling_price", "km_driven", "year"]
for col in [c for c in num_candidates if c in df.columns]:
    plt.figure()
    df[col].hist(bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.show()
```

[4]. Tool: ChatGPT
     Prompt: Write a cleaning and feature engineering cell that converts target and mileage to numeric, drops missing target rows, creates car_age from year, extracts brand from name, and drops the raw name column.
     Output:
```python
# [Use of AI: 4] Cleaning and feature engineering.

df = df.copy()

# Standardize column names (just in case)
df.columns = [c.strip() for c in df.columns]

TARGET = "selling_price"
if TARGET not in df.columns:
    raise ValueError(f"Expected target column '{TARGET}' not found. Found columns: {df.columns.tolist()}")

def to_number(s):
    # Handles commas and stray currency symbols if present
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("₹", "", regex=False)
         .str.strip(),
        errors="coerce"
    )

if "km_driven" in df.columns:
    df["km_driven"] = to_number(df["km_driven"])

df[TARGET] = to_number(df[TARGET])

# Drop rows with missing target
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# Feature: car age
if "year" in df.columns:
    current_year = datetime.now().year
    df["car_age"] = current_year - pd.to_numeric(df["year"], errors="coerce")
else:
    df["car_age"] = np.nan

# Feature: brand from name
if "name" in df.columns:
    df["brand"] = (
        df["name"].astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .str.lower()
    )

# Optional: drop raw name to avoid high-cardinality one-hot explosion
DROP_COLS = [c for c in ["name"] if c in df.columns]
df = df.drop(columns=DROP_COLS)

# Drop any rows with nonsense values (light-touch)
if "km_driven" in df.columns:
    df = df[df["km_driven"].notna()]
    df = df[df["km_driven"] >= 0]

df = df[df[TARGET] > 0]

print("After cleaning:", df.shape)
df.head()
```

[5]. Tool: ChatGPT
     Prompt: Write code to split the dataset into train/validation/test sets with a fixed random seed.
     Output:
```python
# [Use of AI: 5] Data splitting.

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 15% test, 85% temp
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED
)

# From the remaining 85%, use ~20% as validation (so 0.85*0.20 = 17% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_SEED
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
```

[6]. Tool: ChatGPT
     Prompt: Write a scikit-learn preprocessing pipeline for numeric and categorical features using ColumnTransformer, plus helper functions to build pipelines and evaluate regression metrics.
     Output:
```python
# [Use of AI: 6] Build preprocessing (ColumnTransformer) and utilities.

# Identify numeric and categorical columns automatically
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

def make_pipeline(model):
    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

def eval_regression(model, X_tr, y_tr, X_va, y_va):
    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    return {
        "train_mae": mean_absolute_error(y_tr, pred_tr),
        "train_rmse": rmse(y_tr, pred_tr),
        "train_r2": r2_score(y_tr, pred_tr),
        "val_mae": mean_absolute_error(y_va, pred_va),
        "val_rmse": rmse(y_va, pred_va),
        "val_r2": r2_score(y_va, pred_va),
    }
```

[7]. Tool: ChatGPT
     Prompt: Write training code to fit a baseline model, linear regression, tuned ridge, tuned lasso, random forest, and gradient boosting, then collect validation metrics into a results dataframe sorted by validation MAE.
     Output:
```python
# [Use of AI: 7] Train models + collect results.

results = []

def add_result(name, model, metrics):
    row = {"model": name}
    row.update(metrics)
    results.append(row)

# 1) Baseline
baseline = make_pipeline(DummyRegressor(strategy="median"))
baseline.fit(X_train, y_train)
add_result("DummyRegressor(median)", baseline, eval_regression(baseline, X_train, y_train, X_val, y_val))

# 2) Linear Regression
lin = make_pipeline(LinearRegression())
lin.fit(X_train, y_train)
add_result("LinearRegression", lin, eval_regression(lin, X_train, y_train, X_val, y_val))

# 3) Ridge (tune alpha)
ridge = make_pipeline(Ridge(random_state=RANDOM_SEED))
ridge_params = {"model__alpha": np.logspace(-2, 3, 10)}
ridge_search = GridSearchCV(ridge, ridge_params, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_
add_result(f"Ridge(best alpha={ridge_search.best_params_['model__alpha']:.4g})",
           best_ridge, eval_regression(best_ridge, X_train, y_train, X_val, y_val))

# 4) Lasso (tune alpha)
lasso = make_pipeline(Lasso(max_iter=20000, random_state=RANDOM_SEED))
lasso_params = {"model__alpha": np.logspace(-3, 1, 10)}
lasso_search = GridSearchCV(lasso, lasso_params, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
lasso_search.fit(X_train, y_train)
best_lasso = lasso_search.best_estimator_
add_result(f"Lasso(best alpha={lasso_search.best_params_['model__alpha']:.4g})",
           best_lasso, eval_regression(best_lasso, X_train, y_train, X_val, y_val))

# 5) Random Forest (light tuning-free default)
rf = make_pipeline(RandomForestRegressor(
    n_estimators=400,
    random_state=RANDOM_SEED,
    n_jobs=-1
))
rf.fit(X_train, y_train)
add_result("RandomForest(n=400)", rf, eval_regression(rf, X_train, y_train, X_val, y_val))

# 6) Gradient Boosting
gbr = make_pipeline(GradientBoostingRegressor(random_state=RANDOM_SEED))
gbr.fit(X_train, y_train)
add_result("GradientBoostingRegressor", gbr, eval_regression(gbr, X_train, y_train, X_val, y_val))

results_df = pd.DataFrame(results).sort_values("val_mae")
results_df
```

[8]. Tool: ChatGPT
     Prompt: Write code to select the best model by validation MAE, refit on train+validation data, and evaluate on the held-out test set with MAE, RMSE, and R².
     Output:
```python
# [Use of AI: 8] Refit best model and test evaluation.

# Identify best model by validation MAE
best_name = results_df.iloc[0]["model"]
print("Best (by val MAE):", best_name)

# Map names to trained estimators
trained_models = {
    "DummyRegressor(median)": baseline,
    "LinearRegression": lin,
    [r["model"] for r in results if r["model"].startswith("Ridge(")][0]: best_ridge,
    [r["model"] for r in results if r["model"].startswith("Lasso(")][0]: best_lasso,
    "RandomForest(n=400)": rf,
    "GradientBoostingRegressor": gbr
}
best_model = trained_models[best_name]

# Refit on train+val
X_tr_full = pd.concat([X_train, X_val], axis=0)
y_tr_full = pd.concat([y_train, y_val], axis=0)

best_model.fit(X_tr_full, y_tr_full)

test_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = rmse(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Test MAE : {test_mae:,.2f}")
print(f"Test RMSE: {test_rmse:,.2f}")
print(f"Test R^2 : {test_r2:.4f}")
```

[9]. Tool: ChatGPT
     Prompt: Write diagnostic plotting code to visualize predicted vs actual prices and the residual distribution on the test set.
     Output:
```python
# [Use of AI: 9] Predicted vs actual + residuals.

plt.figure()
plt.scatter(y_test, test_pred, alpha=0.6)
plt.title("Predicted vs Actual (Test Set)")
plt.xlabel("Actual selling price")
plt.ylabel("Predicted selling price")
plt.show()

residuals = (y_test - test_pred)

plt.figure()
plt.hist(residuals, bins=40)
plt.title("Residuals (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("count")
plt.show()
```

[10]. Tool: ChatGPT
     Prompt: Write interpretation code that extracts one-hot feature names, and then displays either top feature importances (tree models) or the largest coefficients (linear models).
     Output:
```python
# [Use of AI: 10] Feature importance / coefficients extraction.

def get_feature_names_from_preprocess(preprocess_obj):
    # Works for scikit-learn >= 1.0
    try:
        return preprocess_obj.get_feature_names_out()
    except Exception:
        # fallback: return indices
        return None

pre = best_model.named_steps["preprocess"]
model = best_model.named_steps["model"]
feature_names = get_feature_names_from_preprocess(pre)

def show_top_features(importances, feature_names, top_n=20, title="Top features"):
    if feature_names is None:
        # fallback: just show numeric indices
        feature_names = np.array([f"f{i}" for i in range(len(importances))])
    else:
        feature_names = np.array(feature_names)

    order = np.argsort(importances)[::-1][:top_n]
    top_feats = feature_names[order]
    top_vals = importances[order]

    plt.figure()
    plt.barh(list(reversed(top_feats)), list(reversed(top_vals)))
    plt.title(title)
    plt.xlabel("importance (higher = more)")
    plt.show()

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    show_top_features(importances, feature_names, top_n=20, title="Top 20 Feature Importances")
elif hasattr(model, "coef_"):
    coefs = model.coef_
    # Show by absolute magnitude
    order = np.argsort(np.abs(coefs))[::-1][:20]
    display(pd.DataFrame({
        "feature": (feature_names[order] if feature_names is not None else order),
        "coef": coefs[order],
        "abs_coef": np.abs(coefs[order])
    }).sort_values("abs_coef", ascending=False))
else:
    print("This model type does not expose feature_importances_ or coef_.")
```

[11]. Tool: ChatGPT
     Prompt: Write code to save the best trained pipeline to a joblib file and demonstrate loading it and predicting on a few rows.
     Output:
```python
# [Use of AI: 11] Save pipeline with joblib.

MODEL_PATH = "best_used_car_price_model.joblib"
joblib.dump(best_model, MODEL_PATH)
print("Saved:", MODEL_PATH)

# Example: load and predict on the first 5 rows of the test set
loaded = joblib.load(MODEL_PATH)
loaded.predict(X_test.head())
```
