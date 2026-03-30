import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

import openml
from ucimlrepo import fetch_ucirepo

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -----------------------------
# Functions (UNCHANGED)
# -----------------------------
def profile_target(y: pd.Series) -> Dict:
    y_nonnull = y.dropna()
    if len(y_nonnull) == 0:
        return {"task_type": "unknown", "n_classes": 0}

    if not pd.api.types.is_numeric_dtype(y_nonnull):
        n_classes = y_nonnull.astype(str).nunique()
        return {"task_type": "binary classification" if n_classes <= 2 else "multiclass classification", "n_classes": n_classes}

    n_unique = y_nonnull.nunique()
    if n_unique <= 10:
        return {"task_type": "binary classification" if n_unique <= 2 else "multiclass classification", "n_classes": n_unique}

    return {"task_type": "regression", "n_classes": n_unique}


def missingness_summary(df: pd.DataFrame) -> Dict:
    return {
        "overall_missing_pct": df.isna().mean().mean() * 100,
        "rows_with_missing_pct": df.isna().any(axis=1).mean() * 100
    }


def clean_dataset(df, drop_duplicates=True, missing_tokens=None):
    df = df.copy()

    if missing_tokens is not None:
        df = df.replace(missing_tokens, np.nan)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].where(
            df[col].isna(),
            df[col].astype(str).str.strip().str.replace(".", "", regex=False)
        )

    duplicates_dropped = 0
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        duplicates_dropped = before - len(df)

    return df, duplicates_dropped


def run_mice_imputation(X, random_state=42, max_iter=3):
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=True,
        skip_complete=True
    )
    X_imp = imputer.fit_transform(X)
    return pd.DataFrame(X_imp, columns=X.columns, index=X.index)


# -----------------------------
# DATASETS
# -----------------------------
def load_datasets():
    datasets = []

    ids = [
        (45551, "Atlas Higgs Boson", "Label", [-999]),
        (46888, "Sepsis", "SepsisLabel", None),
        (46860, "Support", "death", None),
        (46882, "Jigsaw", "target", None),
        (46359, "Fraud", "bad_flag", None),
        (41147, "Albert", "class", None),
        (45553, "FICO", "RiskPerformance", [-9, -8, -7]),
    ]

    for did, name, target, missing_tokens in ids:
        dataset = openml.datasets.get_dataset(did)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.concat([X, y], axis=1)

        datasets.append({
            "name": name,
            "df": df,
            "target": target,
            "missing_tokens": missing_tokens
        })

    return datasets


# -----------------------------
# MAIN
# -----------------------------
def main():
    if len(sys.argv) < 2:
        raise ValueError("Provide dataset index")

    dataset_idx = int(sys.argv[1])
    datasets = load_datasets()

    d = datasets[dataset_idx]

    print(f"Running dataset {dataset_idx}: {d['name']}")

    df_clean, _ = clean_dataset(
        d["df"],
        missing_tokens=d.get("missing_tokens")
    )

    y = df_clean[d["target"]]
    X = df_clean.drop(columns=[d["target"]])
    X_num = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if not X_num.empty:
        X_imp = run_mice_imputation(X_num)
    else:
        X_imp = X_num

    # OUTPUT
    output_dir = f"outputs/{d['name'].replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    df_clean.to_csv(f"{output_dir}/cleaned.csv", index=False)
    X_imp.head().to_csv(f"{output_dir}/imputed_preview.csv", index=False)

    print(f"Finished {d['name']}")


if __name__ == "__main__":
    main()