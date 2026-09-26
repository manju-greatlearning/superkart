import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "Product_Store_Sales_Total"
DROP_COLS = ["Product_Id"]

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_train = train.drop(columns=[TARGET] + DROP_COLS)
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET] + DROP_COLS)
y_test = test[TARGET]

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

models = {
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [8, 12, None]},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    ),
}

best = None
results = []
for name, (estimator, params) in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    grid = GridSearchCV(pipe, params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    preds = grid.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    row = {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "best_params": grid.best_params_}
    results.append(row)
    if best is None or rmse < best["rmse"]:
        best = {**row, "pipeline": grid.best_estimator_}

joblib.dump(best["pipeline"], MODEL_DIR / "superkart_sales_model.joblib")
with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump({"best_model": {k: v for k, v in best.items() if k != "pipeline"}, "all_results": results}, f, indent=2)

print(json.dumps({"best_model": {k: v for k, v in best.items() if k != "pipeline"}}, indent=2))
