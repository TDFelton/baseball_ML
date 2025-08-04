#!/usr/bin/env python
from tqdm import tqdm
from mpi4py import MPI
import pandas as pd
import numpy as np
from itertools import product
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys

# -------------------------------------------
# MPI Setup
# -------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------------------------
# Load Preprocessed Data
# -------------------------------------------
X = pd.read_pickle("/sciclone/home/tdfelton/baseball/X_stage2.pkl")
y = pd.read_pickle("/sciclone/home/tdfelton/baseball/y_stage2.pkl")
imp_df = pd.read_pickle("/sciclone/home/tdfelton/baseball/feature_importances.pkl")
categorical_cols = pd.read_pickle("/sciclone/home/tdfelton/baseball/categorical_cols.pkl")

# -------------------------------------------
# Parameter Grid
# -------------------------------------------
top_N_list = list(np.linspace(10, 260, num=25, dtype=int))
scale_pos_weights = list(np.linspace(1.0, 10.0, num=20))
early_stops = list(np.linspace(10, 1100, num=10, dtype=int))
thresholds = list(np.linspace(0.4, 0.6, num=10))

param_grid = list(product(top_N_list, scale_pos_weights, early_stops, thresholds))
split_grid = np.array_split(param_grid, size)
my_grid = split_grid[rank]

print(f"[Rank {rank}] 💡 Received {len(my_grid)} parameter combinations.", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------------------
# Evaluation Function
# -------------------------------------------
def evaluate_combo(top_N, scale_wt, esr, thresh):
    try:
        # Ensure correct types
        top_N = int(top_N)
        esr = int(esr)

        top_features = imp_df.sort_values(by="Importance", ascending=False)["Feature"].head(top_N).tolist()
        cat_top = [col for col in top_features if col in categorical_cols]
        X_subset = X[top_features]

        fold_f1s = []
        for train_idx, test_idx in skf.split(X_subset, y):
            train_idx = train_idx.astype(int)
            test_idx = test_idx.astype(int)

            X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = CatBoostClassifier(
                verbose=0,
                scale_pos_weight=scale_wt,
                eval_metric='F1',
                early_stopping_rounds=esr
            )
            model.fit(X_train, y_train, cat_features=cat_top)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > thresh).astype(int)
            fold_f1s.append(f1_score(y_test, y_pred))

        return {
            'top_N': top_N,
            'scale_pos_weight': scale_wt,
            'early_stopping_rounds': esr,
            'threshold': thresh,
            'mean_f1': np.mean(fold_f1s)
        }

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error in evaluate_combo: {e}", flush=True)
        return None

# -------------------------------------------
# Execute Grid Search
# -------------------------------------------
my_results = []
for i, (top_N, scale_wt, esr, thresh) in enumerate(my_grid):
    print(f"[Rank {rank}] [{i+1}/{len(my_grid)}] Evaluating: top_N={top_N}, scale_wt={scale_wt:.2f}, esr={esr}, thresh={thresh:.3f}", flush=True)
    result = evaluate_combo(top_N, scale_wt, esr, thresh)
    if result is not None:
        my_results.append(result)
    print(f"[Rank {rank}] Evaluating: top_N={top_N}, scale_wt={scale_wt:.2f}, esr={esr}, thresh={thresh:.3f}", flush=True)
    result = evaluate_combo(top_N, scale_wt, esr, thresh)
    if result is not None:
        my_results.append(result)

# -------------------------------------------
# Gather Results
# -------------------------------------------
all_results = comm.gather(my_results, root=0)

if rank == 0:
    flat_results = [item for sublist in all_results for item in sublist if item is not None]
    print(f"[Rank 0] ✅ Writing {len(flat_results)} results to CSV", flush=True)
    df = pd.DataFrame(flat_results)
    df = df.sort_values(by='mean_f1', ascending=False)
    df.to_csv("/sciclone/home/tdfelton/baseball/gridsearch_results.csv", index=False)
    print(df.head(10))
