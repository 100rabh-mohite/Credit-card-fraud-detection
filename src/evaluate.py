"""Evaluate the selected fraud model once on the untouched holdout test set."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")


class ModelEvaluator:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_artifacts(self):
        required = [
            "model.pkl", "X_test.pkl", "y_test.pkl",
            "threshold.pkl", "feature_names.pkl",
        ]
        missing = [name for name in required if not (MODEL_DIR / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts: {', '.join(missing)}. Run python src/train.py first."
            )

        return (
            joblib.load(MODEL_DIR / "model.pkl"),
            joblib.load(MODEL_DIR / "X_test.pkl"),
            joblib.load(MODEL_DIR / "y_test.pkl"),
            float(joblib.load(MODEL_DIR / "threshold.pkl")),
            joblib.load(MODEL_DIR / "feature_names.pkl"),
        )

    def evaluate(self, model, X_test, y_test, threshold, feature_names):
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "Decision Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-Score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "PR-AUC (Average Precision)": average_precision_score(y_test, probabilities),
        }

        print("\nHOLDOUT TEST PERFORMANCE")
        print("=" * 55)
        for name, value in metrics.items():
            print(f"{name:30s}: {value:.6f}")

        report = classification_report(
            y_test, predictions, target_names=["Normal", "Fraud"], zero_division=0
        )
        print("\nClassification report\n")
        print(report)

        with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as file:
            file.write("HOLDOUT TEST PERFORMANCE\n")
            file.write("=" * 55 + "\n")
            for name, value in metrics.items():
                file.write(f"{name:30s}: {value:.6f}\n")
            file.write("\nClassification report\n")
            file.write(report)

        pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

        cm = confusion_matrix(y_test, predictions)
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Normal", "Fraud"]
        ).plot(values_format="d", cmap="Blues")
        plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, probabilities)
        plt.title("ROC Curve — Holdout Test Set")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve — Holdout Test Set")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=160)
        plt.close()

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

            top = importance.head(10).sort_values("Importance")
            plt.figure(figsize=(8, 5))
            plt.barh(top["Feature"], top["Importance"])
            plt.xlabel("Importance")
            plt.title("Top 10 Feature Importances")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
            plt.close()

        print("\nEvaluation completed. Results saved in outputs/.")

    def run(self):
        self.evaluate(*self.load_artifacts())


if __name__ == "__main__":
    ModelEvaluator().run()
