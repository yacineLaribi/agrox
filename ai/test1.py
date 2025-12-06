import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# -------------------
# LOAD DATA
# -------------------
df = pd.read_csv("dataset/genus_data.csv")

# Replace '.' with NaN
df.replace('.', pd.NA, inplace=True)

# -------------------
# CLEAN NUMERIC FEATURES
# -------------------
numeric_cols = ["HybProp","Hyb_Ratio","C_value","CV_C_value","perc_per","perc_wood","perc_ag","tavg"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# -------------------
# CLEAN CATEGORICAL FEATURES
# -------------------
cat_cols = ["Family","Order","floral_symm","mating_system","repro_syndrome","pollination_syndrome","RedList"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# -------------------
# CREATE TARGETS
# -------------------
df["CanHybridize"] = (df["HybProp"] > 0).astype(int)

# -------------------
# FEATURES
# -------------------
reg_features = ["Hyb_Ratio","C_value","CV_C_value","perc_per","perc_wood","perc_ag","tavg"] + cat_cols
clf_features = ["C_value","CV_C_value","perc_per","perc_wood","perc_ag","tavg"] + cat_cols

X_reg = df[reg_features]
y_reg = df["HybProp"]

X_clf = df[clf_features]
y_clf = df["CanHybridize"]

# -------------------
# TRAIN/TEST SPLIT
# -------------------
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# -------------------
# PREPROCESSING
# -------------------
numeric_features_reg = X_train_reg.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features_reg = X_train_reg.select_dtypes(include=["object"]).columns.tolist()

numeric_features_clf = X_train_clf.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features_clf = X_train_clf.select_dtypes(include=["object"]).columns.tolist()

# Regression pipeline
preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_reg),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features_reg)
    ]
)

reg_pipeline = Pipeline([
    ("preprocessor", preprocessor_reg),
    ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42))
])

reg_pipeline.fit(X_train_reg, y_train_reg)
pred_reg = reg_pipeline.predict(X_test_reg)

print("=== Gradient Boosting Regression (HybProp) ===")
print("MAE:", mean_absolute_error(y_test_reg, pred_reg))
print("MSE:", mean_squared_error(y_test_reg, pred_reg))

# Classification pipeline
preprocessor_clf = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_clf),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features_clf)
    ]
)

# Encode training data first for SMOTE
X_train_clf_encoded = preprocessor_clf.fit_transform(X_train_clf)
X_test_clf_encoded = preprocessor_clf.transform(X_test_clf)

# SMOTE oversampling
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_clf_encoded, y_train_clf)

clf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf_model.fit(X_train_res, y_train_res)
pred_clf = clf_model.predict(X_test_clf_encoded)

print("\n=== Random Forest Hybridization Classifier ===")
print("Accuracy:", accuracy_score(y_test_clf, pred_clf))
print("F1 Score:", f1_score(y_test_clf, pred_clf))
