from src.preprocessing import preprocess_data, save_processed_data
from src.baseline import train_and_evaluate as lr_train
from src.xgb_base import train_and_evaluate as xgb_base_train
from src.xgb_tuned import train_and_evaluate as xgb_tuned_train
from src.rf_base import train_and_evaluate as rf_base_train
from src.rf_tuned import train_and_evaluate as rf_tuned_train
import json
import os

def run_pipeline(csv_path):
    X_train, X_test, y_train, y_test = preprocess_data(csv_path)
    save_processed_data(X_train, X_test, y_train, y_test)

    lr_metrics = lr_train(X_train, X_test, y_train, y_test)
    xgb_base_metrics = xgb_base_train(X_train, X_test, y_train, y_test)
    xgb_tuned_metrics, xgb_best_params = xgb_tuned_train(X_train, X_test, y_train, y_test)
    rf_base_metrics = rf_base_train(X_train, X_test, y_train, y_test)
    rf_tuned_metrics, rf_best_params = rf_tuned_train(X_train, X_test, y_train, y_test)

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/metrics.json", "w") as f:
        json.dump(
            {
                "logistic_regression": lr_metrics,
                "xgb_base": xgb_base_metrics,
                "xgb_tuned": xgb_tuned_metrics,
                "xgb_best_params": xgb_best_params,
                "rf_base": rf_base_metrics,
                "rf_tuned": rf_tuned_metrics,
                "rf_best_params": rf_best_params
            },
            f,
            indent=4
        )
