from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_base.pkl")

    return metrics
