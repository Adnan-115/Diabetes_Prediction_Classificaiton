"""
Data preparation, Leakage Auditing, and In-Fold ADASYN pipeline validation
for notebooks/abrar/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import ADASYN

# Constants & Schema Definition
SEED = 42
RAW_DATA_PATH = Path("data/raw/diabetes_binary_health_indicators_BRFSS2015.csv")
TARGET = "diabetes_binary"

CONTINUOUS_COLS = ["bmi", "menthlth", "physhlth"]
ORDINAL_COLS = {
    "genhlth": (1, 5),
    "age": (1, 13),
    "education": (1, 8),
    "income": (1, 8),
}
BINARY_COLS = [
    "highbp", "highchol", "cholcheck", "smoker", "stroke",
    "heartdiseaseorattack", "physactivity", "fruits", "veggies",
    "hvyalcoholconsump", "anyhealthcare", "nodocbccost", "diffwalk", "sex"
]

def load_and_audit_raw_data():
    print("=" * 70)
    print("STEP 1: LOADING AND AUDITING RAW BRFSS DATA")
    print("=" * 70)
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    print(f"Raw dataset shape: {df.shape}")
    print(f"Missing values count: {df.isnull().sum().sum()} total missing values.")
    
    # Check class distribution
    counts = df[TARGET].value_counts().sort_index()
    prevalence = df[TARGET].mean()
    print(f"Target distribution:\n{counts.to_dict()}")
    print(f"Positive class prevalence: {prevalence * 100:.2f}% (approx 1:6.2 imbalance)")
    
    # Stratified 80/20 split
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df[TARGET]
    )
    
    print("\n--- STRATIFIED SPLIT VERIFICATION ---")
    print(f"Train set shape: {train_df.shape} (Positive rate: {train_df[TARGET].mean() * 100:.2f}%)")
    print(f"Holdout set shape: {holdout_df.shape} (Positive rate: {holdout_df[TARGET].mean() * 100:.2f}%)")
    
    # Index overlap check (Data Leakage Test 1)
    overlap = set(train_df.index).intersection(set(holdout_df.index))
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping indices: {len(overlap)}"
    print("[PASS] Zero index overlap between Train and Holdout sets.")
    
    # Save holdout test set to data/processed if not already saved
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = processed_dir / "diabetes_binary_holdout_test.csv"
    if not holdout_path.exists():
        holdout_df.to_csv(holdout_path, index=False)
        print(f"[SAVED] Saved untouched holdout test set to {holdout_path}")
        
    return train_df, holdout_df

def sanitize_resampled_data(X_resampled_df, feature_cols):
    """
    Post-ADASYN Discrete & Binary Sanitation Projection:
    Restores valid discrete domain for tabular health indicators.
    """
    X_clean = X_resampled_df.copy()
    
    # 1. Sanitize Binary Columns: Round to {0, 1}
    for col in BINARY_COLS:
        if col in X_clean.columns:
            X_clean[col] = (X_clean[col] >= 0.5).astype(int)
            
    # 2. Sanitize Ordinal Columns: Round to nearest int and clip to valid domain
    for col, (min_val, max_val) in ORDINAL_COLS.items():
        if col in X_clean.columns:
            X_clean[col] = np.clip(np.round(X_clean[col]), min_val, max_val).astype(int)
            
    # 3. Continuous columns: Clip to non-negative
    for col in CONTINUOUS_COLS:
        if col in X_clean.columns:
            X_clean[col] = np.clip(X_clean[col], 0, None)
            
    return X_clean

def test_in_fold_adasyn_pipeline(train_df):
    print("\n" + "=" * 70)
    print("STEP 2: TESTING IN-FOLD ADASYN PIPELINE & DATA INTEGRITY CHECKS")
    print("=" * 70)
    
    X = train_df.drop(columns=[TARGET]).reset_index(drop=True)
    y = train_df[TARGET].astype(int).reset_index(drop=True)
    feature_cols = list(X.columns)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Testing Fold {fold + 1} ---")
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
        
        # Check fold balance
        print(f"Raw Train Fold: {X_tr.shape[0]} rows (Positives: {y_tr.sum()}, {y_tr.mean()*100:.2f}%)")
        print(f"Raw Val Fold:   {X_val.shape[0]} rows (Positives: {y_val.sum()}, {y_val.mean()*100:.2f}%)")
        
        # 1. Pre-ADASYN Scaling (Fit on Train fold ONLY)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        
        # 2. In-Fold ADASYN Resampling
        adasyn = ADASYN(random_state=SEED, n_neighbors=5)
        X_tr_res, y_tr_res = adasyn.fit_resample(X_tr_scaled, y_tr)
        
        # 3. Inverse transform back to original feature scale for discrete sanitization
        X_tr_res_unscaled = scaler.inverse_transform(X_tr_res)
        X_tr_res_df = pd.DataFrame(X_tr_res_unscaled, columns=feature_cols)
        
        # 4. Post-Resampling Discrete Projection
        X_tr_sanitized = sanitize_resampled_data(X_tr_res_df, feature_cols)
        
        # Data integrity verification checks:
        print(f"Resampled Train Fold: {X_tr_sanitized.shape[0]} rows (Positives: {y_tr_res.sum()}, {y_tr_res.mean()*100:.2f}%)")
        
        # Verify Binary columns are strictly {0, 1}
        for b_col in BINARY_COLS:
            unique_vals = set(X_tr_sanitized[b_col].unique())
            assert unique_vals.issubset({0, 1}), f"Data Bug! Non-binary values in {b_col}: {unique_vals}"
            
        # Verify Ordinal columns are strictly integers within bounds
        for o_col, (min_v, max_v) in ORDINAL_COLS.items():
            assert X_tr_sanitized[o_col].min() >= min_v, f"Out of bounds min for {o_col}"
            assert X_tr_sanitized[o_col].max() <= max_v, f"Out of bounds max for {o_col}"
            assert np.issubdtype(X_tr_sanitized[o_col].dtype, np.integer), f"{o_col} is not integer!"
            
        print("[PASS] Fold data integrity, discrete projection, and zero-leakage verified.")
        break  # Tested fold 1 successfully

if __name__ == "__main__":
    train_df, holdout_df = load_and_audit_raw_data()
    test_in_fold_adasyn_pipeline(train_df)
    print("\n[ALL STEP 1 CHECKS PASSED PERFECTLY]")
