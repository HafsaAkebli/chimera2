# clinical/one_hot_encode.py
# Build a single-case clinical feature vector using the training meta:
# - Standardize fields like in training
# - Categorical -> one-hot with guaranteed "Missing"
# - Numeric -> median from meta, then z-score using mean/std from meta
# Returns: (X, colnames) where X is np.ndarray shape (1, D), dtype float32

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Column sets used during training
CATEGORICAL = [
    "sex","smoking","tumor","stage","substage","grade",
    "reTUR","LVI","variant","EORTC"
]
NUMERIC = ["age","no_instillations"]


def _standardize_like_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace -1 and other invalids with NaN
    df.replace([-1, "-1", "nan", "NaN", "None", ""], np.nan, inplace=True)

    # String normalizations (match training)
    if "sex" in df:
        df["sex"] = df["sex"].astype(str).str.lower().map(
            {"m": "Male", "f": "Female", "male": "Male", "female": "Female"}
        )
    if "smoking" in df:
        df["smoking"] = df["smoking"].astype(str).str.capitalize()
    if "tumor" in df:
        df["tumor"] = df["tumor"].astype(str).str.capitalize()
    if "reTUR" in df:
        df["reTUR"] = df["reTUR"].replace({"yes": "Yes", "no": "No"})
    if "LVI" in df:
        df["LVI"] = df["LVI"].replace({"yes": "Yes", "no": "No"})

    return df


def encode_patient(patient_data: dict, meta_path: str):
    """
    Args:
        patient_data: dict loaded from the challenge JSON for a single case
        meta_path: path to clinical_preproc_meta_T2.json saved during training

    Returns:
        X: np.ndarray with shape (1, D), dtype float32
        columns: list[str] with the exact column order used to build X
    """
    # Load meta (categorical categories + numeric stats)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # One-row dataframe
    df = pd.DataFrame([patient_data])
    df = _standardize_like_training(df)

    # ------ CATEGORICAL -> one-hot with guaranteed columns/order ------
    expected_onehot_cols = []
    for c in CATEGORICAL:
        if c in meta.get("categorical", {}):
            categories = meta["categorical"][c]
            expected_onehot_cols += [f"{c}_{k}" for k in categories]

    # Ensure each categorical has a value; fallback to "Missing"
    cat_df = pd.DataFrame(index=[0])
    for c in CATEGORICAL:
        if c in meta.get("categorical", {}):
            val = df.get(c, pd.Series([np.nan])).iloc[0]
            if pd.isna(val) or val not in meta["categorical"][c]:
                val = "Missing"
            cat_df[c] = str(val)

    # One-hot and reindex to expected columns
    if not cat_df.empty:
        cat_oh = pd.get_dummies(cat_df, prefix=CATEGORICAL, columns=CATEGORICAL, dtype=np.uint8)
        cat_oh = cat_oh.reindex(columns=expected_onehot_cols, fill_value=0).astype(np.uint8)
    else:
        cat_oh = pd.DataFrame(columns=expected_onehot_cols, data=[[0]*len(expected_onehot_cols)], dtype=np.uint8)

    # ------ NUMERIC -> impute median, then z-score using meta ------
    num_vals = []
    for c in NUMERIC:
        stats = meta.get("numeric", {}).get(c, None)
        raw = df.get(c, pd.Series([np.nan])).iloc[0]
        try:
            raw = float(raw)
        except Exception:
            raw = np.nan

        if stats is None:
            num_z = 0.0
        else:
            median = stats.get("median", 0.0)
            mean   = stats.get("mean", 0.0)
            std    = stats.get("std", 1.0) or 1.0
            val = median if (raw is None or np.isnan(raw)) else raw
            num_z = (val - mean) / (std if std != 0 else 1.0)
        num_vals.append(num_z)

    num_cols = NUMERIC[:]

    # ------ Concatenate ------
    X_cat = cat_oh.values.astype(np.float32)
    X_num = np.asarray(num_vals, dtype=np.float32)[None, :]
    X = np.concatenate([X_cat, X_num], axis=1)
    columns = list(cat_oh.columns) + num_cols

    print(f"✅ Clinical one-hot+numeric vector built. Shape: {X.shape}")
    return X, columns




#PATIENT_JSON = Path("/mnt/dmif-nas/MITEL/hafsa/chimera_bcg/Task2/input1/chimera-clinical-data-of-bladder-cancer-patients.json")
#META_PATH = Path("/mnt/dmif-nas/MITEL/hafsa/chimera_bcg/Task2/model3/clinical/clinical_preproc_meta_T2.json")

# === load single patient data ===
#with open(PATIENT_JSON, "r") as f:
    #patient_data = json.load(f)

# === encode ===
#X, cols = encode_patient(patient_data, str(META_PATH))

# === print results ===
#print(f"   Shape: {X.shape}\n")
#print("X =", X)

#for name, val in zip(cols, X[0]):
    #print(f"{name:20s} {val:.3f}")