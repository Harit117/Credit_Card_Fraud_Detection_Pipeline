from src.train import run_pipeline
from src.evaluate import tune_threshold

if __name__ == "__main__":
    run_pipeline("data/raw/creditcard.csv")

    tune_threshold(
        model_path="models/rf_tuned.pkl",
        X_test_path="data/processed/X_test.csv",
        y_test_path="data/processed/y_test.csv",
        target_recall=0.95
    )
