"""Train and select a fraud classifier without leaking test information.

Workflow:
1. Split the original data into train / validation / test sets.
2. Fit the scaler on the training set only.
3. Apply SMOTE to the training set only.
4. Select the model and decision threshold using validation data.
5. Keep the test set untouched until final evaluation.
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
DATA_PATH = Path("data/creditcard_clean.csv")
MODEL_DIR = Path("models")


class ModelTrainer:
    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load data and validate the target and feature columns."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}. "
                "Read the Dataset section in README.md before training."
            )

        df = pd.read_csv(DATA_PATH)

        if "Class" not in df.columns:
            raise ValueError("The dataset must contain a 'Class' target column.")

        if df["Class"].isna().any():
            raise ValueError("The target column 'Class' contains missing values.")

        if set(df["Class"].unique()) != {0, 1}:
            raise ValueError("'Class' must contain exactly the labels 0 (normal) and 1 (fraud).")

        X = df.drop(columns="Class")
        y = df["Class"].astype("int8")

        if X.isna().any().any():
            raise ValueError("Feature columns contain missing values. Clean them before training.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("All model feature columns must be numeric.")

        print(f"Loaded {len(df):,} transactions and {X.shape[1]} features.")
        print("Original class distribution:")
        print(y.value_counts().rename(index={0: "Normal", 1: "Fraud"}))

        return X, y

    def split_and_preprocess(self, X, y):
        """Create 60/20/20 splits; fit preprocessing on training data only."""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        print("\nApplying SMOTE to training data only...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(
            X_train_scaled, y_train
        )

        print(f"Training rows before SMOTE: {len(y_train):,}")
        print(f"Training rows after SMOTE:  {len(y_train_balanced):,}")
        print(
            "Split sizes — "
            f"train: {len(y_train):,}, validation: {len(y_valid):,}, "
            f"test: {len(y_test):,}"
        )

        return (
            X_train_balanced,
            y_train_balanced,
            X_valid_scaled,
            y_valid,
            X_test_scaled,
            y_test,
            scaler,
            list(X.columns),
        )

    @staticmethod
    def build_models():
        """Use conservative parallelism to reduce memory pressure."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=2,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
        }

    @staticmethod
    def find_best_threshold(y_true, probabilities):
        """Choose the threshold with the best validation F1 score."""
        precision, recall, thresholds = precision_recall_curve(
            y_true, probabilities
        )

        if len(thresholds) == 0:
            return 0.5, 0.0

        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_index = int(f1_values.argmax())

        return float(thresholds[best_index]), float(f1_values[best_index])

    def train_and_select(
        self, X_train, y_train, X_valid, y_valid
    ):
        results = []
        trained_models = {}
        thresholds = {}

        print("\nTraining models and selecting thresholds on validation data...")
        for name, model in self.build_models().items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            valid_probabilities = model.predict_proba(X_valid)[:, 1]
            threshold, valid_f1 = self.find_best_threshold(
                y_valid, valid_probabilities
            )
            valid_predictions = (valid_probabilities >= threshold).astype(int)

            trained_models[name] = model
            thresholds[name] = threshold
            results.append(
                {
                    "Model": name,
                    "Validation Threshold": round(threshold, 6),
                    "Validation Precision": round(
                        precision_score(y_valid, valid_predictions, zero_division=0), 4
                    ),
                    "Validation Recall": round(
                        recall_score(y_valid, valid_predictions, zero_division=0), 4
                    ),
                    "Validation F1": round(valid_f1, 4),
                    "Validation ROC-AUC": round(
                        roc_auc_score(y_valid, valid_probabilities), 4
                    ),
                }
            )

        results_df = pd.DataFrame(results).sort_values(
            "Validation F1", ascending=False
        )
        results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)
        print("\nValidation model comparison:")
        print(results_df.to_string(index=False))

        best_name = results_df.iloc[0]["Model"]
        print(f"\nSelected model: {best_name}")
        print(f"Selected threshold: {thresholds[best_name]:.6f}")

        return trained_models[best_name], thresholds[best_name], best_name

    @staticmethod
    def save_artifacts(
        model, scaler, threshold, feature_names, X_test, y_test, model_name
    ):
        """Save everything required for consistent evaluation and inference."""
        artifacts = {
            "model.pkl": model,
            "scaler.pkl": scaler,
            "threshold.pkl": threshold,
            "feature_names.pkl": feature_names,
            "X_test.pkl": X_test,
            "y_test.pkl": y_test,
            "model_name.pkl": model_name,
        }

        for filename, artifact in artifacts.items():
            joblib.dump(artifact, MODEL_DIR / filename, compress=3)

        print("\nSaved model, scaler, threshold, feature names, and holdout test data.")

    def run(self):
        X, y = self.load_data()
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            scaler,
            feature_names,
        ) = self.split_and_preprocess(X, y)

        model, threshold, model_name = self.train_and_select(
            X_train, y_train, X_valid, y_valid
        )
        self.save_artifacts(
            model,
            scaler,
            threshold,
            feature_names,
            X_test,
            y_test,
            model_name,
        )


if __name__ == "__main__":
    ModelTrainer().run()
