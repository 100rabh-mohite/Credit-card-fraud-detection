"""Load the saved pipeline artifacts and predict one transaction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")


class FraudPredictor:
    def __init__(self):
        required = [
            "model.pkl", "scaler.pkl", "threshold.pkl", "feature_names.pkl"
        ]
        missing = [name for name in required if not (MODEL_DIR / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts: {', '.join(missing)}. Train the model first."
            )

        self.model = joblib.load(MODEL_DIR / "model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))
        self.feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    def predict(self, transaction):
        """Return class, fraud probability, and the applied threshold."""
        if not isinstance(transaction, dict):
            raise TypeError("transaction must be a dictionary of feature values.")

        missing = sorted(set(self.feature_names) - set(transaction))
        extra = sorted(set(transaction) - set(self.feature_names))
        if missing or extra:
            raise ValueError(
                f"Feature mismatch. Missing: {missing or 'none'}; "
                f"unexpected: {extra or 'none'}."
            )

        data = pd.DataFrame(
            [[transaction[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        data = data.apply(pd.to_numeric, errors="raise")

        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError("All feature values must be finite numbers.")

        scaled = self.scaler.transform(data)
        probability = float(self.model.predict_proba(scaled)[0, 1])
        prediction = int(probability >= self.threshold)
        label = "Fraud Transaction" if prediction else "Normal Transaction"

        return label, probability, self.threshold


if __name__ == "__main__":
    predictor = FraudPredictor()

    # Example only: replace these values with a real row from your dataset.
    sample_transaction = {name: 0.0 for name in predictor.feature_names}
    sample_transaction["Amount"] = 50.0

    result, probability, threshold = predictor.predict(sample_transaction)
    print(f"Prediction: {result}")
    print(f"Fraud probability: {probability:.4%}")
    print(f"Decision threshold: {threshold:.4f}")
