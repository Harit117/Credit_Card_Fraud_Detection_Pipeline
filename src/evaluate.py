import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def tune_threshold(model_path, X_test_path, y_test_path, target_recall=0.95):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    thresholds = np.linspace(0.01, 0.9, 100)
    candidates = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        if recall >= target_recall:
            candidates.append((t, precision, recall))

    best_threshold, best_precision, best_recall = max(candidates, key=lambda x: x[1])

    results = {
        "roc_auc": roc_auc,
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/threshold_tuned_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
