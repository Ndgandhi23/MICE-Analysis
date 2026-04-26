"""
For each dataset, this script:
- cleans missing tokens
- subsamples the datasets with balanced classes
- normalizes numeric features
- saves outputs as CSVs
Make sure to run download_datasets.py first to access all the datasets
"""

import numpy as np
import pandas as pd
import os

os.makedirs("preprocessed", exist_ok=True)

RANDOM_STATE = 42

# Minority class is preserved up to its natural size.
SUBSAMPLE_MAX = {
    "Atlas Higgs Boson":       50_000,
    "Support":                 10_000,
    "Jigsaw Unintended Bias":  50_000,
    "Fraud Detection":         20_000,
    "Albert":                  20_000,
    "FICO HELOC":              10_000,
    "Wine Quality":             5_000,
    "Diabetes (Pima)":          5_000,
    "Banknote Authentication":  5_000,
}

MIN_CLASS_FRAC = 0.20   


def clean_missing_tokens(df, tokens):
    if tokens:
        df = df.replace(tokens, np.nan)
    return df


def fix_diabetes_zeros(df):
    df = df.copy()
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        matched = [c for c in df.columns if c.lower() == col.lower()]
        for c in matched:
            n = (df[c] == 0).sum()
            if n > 0:
                df[c] = df[c].replace(0, np.nan)
                print(f"  [{col}] replaced {n} zeros -> NaN")
    return df


def binarize_wine_quality(df, target_col="Class", threshold=4):
    #Binarizes wine quality properly
    df = df.copy()
    q = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = (q >= threshold).astype(int)
    n, t = int(df[target_col].sum()), len(df)
    print(f"  Binarized at >={threshold}: {n} good ({100*n/t:.1f}%) | {t-n} poor ({100*(t-n)/t:.1f}%)")
    return df


def introduce_mcar(df, missing_fraction, protect_cols=None, random_state=42):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in (protect_cols or [])]
    for col in cols:
        idx = rng.choice(len(df), size=int(np.ceil(missing_fraction * len(df))), replace=False)
        df.iloc[idx, df.columns.get_loc(col)] = np.nan
    print(f"  MCAR {missing_fraction*100:.0f}% applied across {len(cols)} feature columns")
    return df


def normalize_01(df, protect_cols=None):
    df = df.copy()
    skip = set(protect_cols or [])
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in skip:
            continue
        obs = df[col].dropna()
        if obs.empty:
            continue
        lo, hi = obs.min(), obs.max()
        if lo == hi:
            df[col] = 0.0
        else:
            df[col] = (df[col] - lo) / (hi - lo)
    return df


def stratified_subsample(df, target_col, max_rows, min_class_frac=MIN_CLASS_FRAC,
                          random_state=RANDOM_STATE):
    if len(df) <= max_rows:
        print(f"  No subsampling needed ({len(df)} rows <= {max_rows})")
        return df.copy()

    y = df[target_col].dropna().astype(str)
    is_classification = y.nunique() <= 20

    if not is_classification:
        # Regression: simple random sample
        result = df.sample(n=max_rows, random_state=random_state)
        print(f"  Regression subsample: {len(df)} -> {len(result)}")
        return result

    classes = y.value_counts()
    n_classes = len(classes)

    min_per_class = max(1, int(min_class_frac * max_rows))
    proportional = (classes / classes.sum() * max_rows).astype(int)
    per_class = proportional.clip(lower=min_per_class)

    total = per_class.sum()
    if total > max_rows:
        scale = max_rows / total
        per_class = (per_class * scale).astype(int).clip(lower=1)

    sampled_parts = []
    for cls, n in per_class.items():
        subset = df[df[target_col].astype(str) == cls]
        n = min(n, len(subset))
        sampled_parts.append(subset.sample(n=n, random_state=random_state))

    result = pd.concat(sampled_parts).sample(frac=1, random_state=random_state)  # shuffle

    print(f"  Stratified subsample: {len(df)} -> {len(result)}")
    for cls, n in result[target_col].astype(str).value_counts().items():
        print(f"    class '{cls}': {n} ({100*n/len(result):.1f}%)")

    return result


def preprocess_and_save(name, df, target_col,
                         missing_tokens=None,
                         binarize_wine=False,
                         introduce_mcar_frac=None,
                         diabetes_zero_fix=False,
                         out_filename=None):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  shape: {df.shape}  |  target: {target_col}")
    print(f"{'='*60}")

    df = df.copy()

    if missing_tokens:
        df = clean_missing_tokens(df, missing_tokens)
        print(f"  Replaced missing tokens {missing_tokens} with NaN")

    if diabetes_zero_fix:
        df = fix_diabetes_zeros(df)

    if binarize_wine:
        df = binarize_wine_quality(df, target_col=target_col, threshold=4)

    if introduce_mcar_frac is not None:
        df = introduce_mcar(df, missing_fraction=introduce_mcar_frac,
                            protect_cols=[target_col], random_state=RANDOM_STATE)

    df = stratified_subsample(df, target_col=target_col,
                              max_rows=SUBSAMPLE_MAX[name])

    df = normalize_01(df, protect_cols=[target_col])
    print(f"  Normalized numeric features to [0, 1]")

    overall_miss = df.isna().mean().mean() * 100
    rows_miss    = df.isna().any(axis=1).mean() * 100
    print(f"  Final shape: {df.shape}")
    print(f"  Overall missing: {overall_miss:.2f}% | Rows with missing: {rows_miss:.2f}%")

    fname = out_filename or name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".csv"
    out_path = os.path.join("preprocessed", fname)
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    return df

df_atlas    = pd.read_csv("datasets/df_atlas.csv")
df_support  = pd.read_csv("datasets/df_support.csv")
df_jigsaw   = pd.read_csv("datasets/df_jigsaw.csv")
df_fraud    = pd.read_csv("datasets/df_fraud.csv")
df_albert   = pd.read_csv("datasets/df_albert.csv")
df_fico     = pd.read_csv("datasets/df_fico.csv")
df_wine     = pd.read_csv("datasets/df_wine.csv")
df_diabetes = pd.read_csv("datasets/df_diabetes.csv")
df_banknote = pd.read_csv("datasets/df_banknote.csv")


preprocess_and_save(
    name="Atlas Higgs Boson",
    df=df_atlas,
    target_col="Label",
    missing_tokens=[-999],
    out_filename="atlas_higgs_boson.csv",
)

preprocess_and_save(
    name="Support",
    df=df_support,
    target_col="death",
    out_filename="support.csv",
)

preprocess_and_save(
    name="Jigsaw Unintended Bias",
    df=df_jigsaw,
    target_col="target",
    out_filename="jigsaw_unintended_bias.csv",
)

preprocess_and_save(
    name="Fraud Detection",
    df=df_fraud,
    target_col="bad_flag",
    out_filename="fraud_detection.csv",
)

preprocess_and_save(
    name="Albert",
    df=df_albert,
    target_col="class",
    out_filename="albert.csv",
)

preprocess_and_save(
    name="FICO HELOC",
    df=df_fico,
    target_col="RiskPerformance",
    missing_tokens=[-9, -8, -7],
    out_filename="fico_heloc.csv",
)

preprocess_and_save(
    name="Wine Quality",
    df=df_wine,
    target_col="Class",
    binarize_wine=True,         
    introduce_mcar_frac=0.10,    
    out_filename="wine_quality.csv",
)

preprocess_and_save(
    name="Diabetes (Pima)",
    df=df_diabetes,
    target_col="Outcome",
    diabetes_zero_fix=True,    
    introduce_mcar_frac=0.05,    
    out_filename="diabetes_pima.csv",
)

preprocess_and_save(
    name="Banknote Authentication",
    df=df_banknote,
    target_col="Class",
    introduce_mcar_frac=0.10, 
    out_filename="banknote_authentication.csv",
)

print("\n\nAll datasets preprocessed and saved to preprocessed/")
print("="*60)

#Summary Table
summary_rows = []
for fname in sorted(os.listdir("preprocessed")):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join("preprocessed", fname))
    summary_rows.append({
        "File":             fname,
        "Rows":             len(df),
        "Cols":             len(df.columns),
        "Missing (%)":      round(df.isna().mean().mean() * 100, 2),
    })

summary = pd.DataFrame(summary_rows)
print("\nSummary of preprocessed outputs:")
print(summary.to_string(index=False))