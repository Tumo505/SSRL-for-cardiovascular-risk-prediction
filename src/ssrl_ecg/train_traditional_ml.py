from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices


def extract_ecg_features(signal: np.ndarray, sampling_rate: int = 100) -> dict:
    """Extract handcrafted ECG features from a single 12-lead record."""
    features = {}

    # Per-lead statistics (simplified: use first lead for now)
    lead = signal[0] if signal.ndim > 1 else signal
    features["mean"] = float(np.mean(lead))
    features["std"] = float(np.std(lead))
    features["max"] = float(np.max(lead))
    features["min"] = float(np.min(lead))
    features["range"] = float(np.ptp(lead))

    # Derivative-based features (rough QRS detection proxy)
    diff = np.diff(lead)
    features["mean_gradient"] = float(np.mean(np.abs(diff)))
    features["max_gradient"] = float(np.max(np.abs(diff)))

    # Energy and entropy proxies
    features["energy"] = float(np.sum(lead**2))
    features["entropy"] = float(-np.sum((lead**2 / np.sum(lead**2 + 1e-10)) * np.log(lead**2 / np.sum(lead**2 + 1e-10) + 1e-10)))

    # Heart rate estimate (zero crossings as simple proxy)
    zero_crossings = np.sum(np.diff(np.sign(lead - np.mean(lead))) != 0)
    features["zero_crossings"] = float(zero_crossings)

    # RMS energy per lead (if multi-lead, compute for each)
    if signal.ndim > 1:
        for i in range(min(12, signal.shape[0])):
            l = signal[i]
            features[f"rms_lead_{i}"] = float(np.sqrt(np.mean(l**2)))
    else:
        features["rms"] = float(np.sqrt(np.mean(lead**2)))

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional ML baseline: handcrafted features + RF/XGBoost.")
    parser.add_argument("--data-root", type=Path, default=Path("data/PTB-XL"))
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["rf", "xgb"], default="rf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    db_df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(db_df)
    sampled_train_idx = sample_labelled_indices(splits.train_idx, labels, args.label_fraction, args.seed)

    print("Extracting training features...")
    train_ds = PTBXLRecordDataset(args.data_root, db_df, labels, sampled_train_idx, return_labels=True)
    train_features = []
    train_labels = []
    for i in tqdm(range(len(train_ds)), desc="Train"):
        x, y = train_ds[i]
        feat_dict = extract_ecg_features(x.numpy())
        train_features.append(feat_dict)
        train_labels.append(y.numpy())

    print("Extracting validation features...")
    val_ds = PTBXLRecordDataset(args.data_root, db_df, labels, splits.val_idx, return_labels=True)
    val_features = []
    val_labels = []
    for i in tqdm(range(len(val_ds)), desc="Val"):
        x, y = val_ds[i]
        feat_dict = extract_ecg_features(x.numpy())
        val_features.append(feat_dict)
        val_labels.append(y.numpy())

    print("Extracting test features...")
    test_ds = PTBXLRecordDataset(args.data_root, db_df, labels, splits.test_idx, return_labels=True)
    test_features = []
    test_labels = []
    for i in tqdm(range(len(test_ds)), desc="Test"):
        x, y = test_ds[i]
        feat_dict = extract_ecg_features(x.numpy())
        test_features.append(feat_dict)
        test_labels.append(y.numpy())

    X_train = pd.DataFrame(train_features).fillna(0)
    X_val = pd.DataFrame(val_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)
    y_train = np.vstack(train_labels)
    y_val = np.vstack(val_labels)
    y_test = np.vstack(test_labels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Feature shape: {X_train.shape}")
    print(f"Training on {len(X_train)} samples from label fraction {args.label_fraction}")

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=args.seed, n_jobs=-1)
    else:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=args.seed, n_jobs=-1)
        except ImportError:
            print("XGBoost not installed; falling back to RandomForest.")
            model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=args.seed, n_jobs=-1)

    print(f"Training {args.model.upper()} classifier...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)

    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)

    def multilabel_eval(y_true, y_pred, y_prob):
        results = {}
        # Convert list of arrays to single array for multi-output sklearn models
        if isinstance(y_prob, list):
            y_prob = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.squeeze() for p in y_prob])
        
        for c in range(y_true.shape[1]):
            yc = y_true[:, c]
            if len(np.unique(yc)) < 2:
                continue
            prob_c = y_prob[:, c] if y_prob.ndim == 2 else y_prob
            auc = roc_auc_score(yc, prob_c)
            results[f"auroc_c{c}"] = float(auc)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results["f1_macro"] = float(f1)
        return results

    print("Val metrics:", multilabel_eval(y_val, y_pred_val, y_prob_val))
    print("Test metrics:", multilabel_eval(y_test, y_pred_test, y_prob_test))


if __name__ == "__main__":
    main()
