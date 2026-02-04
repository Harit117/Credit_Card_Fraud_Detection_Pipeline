from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test):
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    grid = GridSearchCV(
        base_model,
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgb_tuned.pkl")

    return metrics, grid.best_params_
