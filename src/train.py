import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


class ModelTrainer:

    def __init__(self):

        self.data_path = "data/creditcard_clean.csv"

        self.model_dir = "models"

        self.model_path = "models/model.pkl"
        self.scaler_path = "models/scaler.pkl"

        self.x_test_path = "models/X_test.pkl"
        self.y_test_path = "models/y_test.pkl"

        self.results_path = "models/model_results.csv"

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )


    def load_data(self):

        print("Loading cleaned dataset...")

        df = pd.read_csv(
            self.data_path
        )

        X = df.drop(
            "Class",
            axis=1
        )

        y = df["Class"]

        return X, y



    def preprocess(self, X, y):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )


        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )


        print("\nApplying SMOTE...")

        smote = SMOTE(
            random_state=42
        )


        X_train, y_train = smote.fit_resample(
            X_train,
            y_train
        )


        print("SMOTE Applied Successfully.")


        return (
            X_train,
            X_test,
            y_train,
            y_test,
            scaler
        )



    def train_models(self, X_train, y_train):

        print("\nTraining Models...")


        models = {

            "Logistic Regression":

            LogisticRegression(
                max_iter=3000,
                solver="lbfgs"
            ),


            "Decision Tree":

            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),


            "Random Forest":

            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),


            "XGBoost":

            XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
        }


        trained_models = {}


        for name, model in models.items():

            print(
                f"Training {name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            trained_models[name] = model


        return trained_models



    def evaluate_models(
            self,
            models,
            X_test,
            y_test
    ):

        results = []


        print(
            "\n================ MODEL COMPARISON ================"
        )


        for name, model in models.items():

            prediction = model.predict(
                X_test
            )

            probability = model.predict_proba(
                X_test
            )[:,1]


            results.append({

                "Model": name,

                "Accuracy":
                round(
                    accuracy_score(
                        y_test,
                        prediction
                    ),
                    4
                ),


                "Precision":
                round(
                    precision_score(
                        y_test,
                        prediction
                    ),
                    4
                ),


                "Recall":
                round(
                    recall_score(
                        y_test,
                        prediction
                    ),
                    4
                ),


                "F1-Score":
                round(
                    f1_score(
                        y_test,
                        prediction
                    ),
                    4
                ),


                "ROC-AUC":
                round(
                    roc_auc_score(
                        y_test,
                        probability
                    ),
                    4
                )

            })


        result_df = pd.DataFrame(
            results
        )


        print(result_df)


        result_df.to_csv(
            self.results_path,
            index=False
        )


        best_model_name = (

            result_df
            .sort_values(
                "F1-Score",
                ascending=False
            )
            .iloc[0]["Model"]

        )


        best_score = (

            result_df
            .sort_values(
                "F1-Score",
                ascending=False
            )
            .iloc[0]["F1-Score"]

        )


        print(
            f"\nBest Model : {best_model_name}"
        )

        print(
            f"Best F1 Score : {best_score}"
        )


        return models[best_model_name]



    def save_files(
            self,
            model,
            scaler,
            X_test,
            y_test
    ):


        print(
            "\nSaving files..."
        )


        joblib.dump(
            model,
            self.model_path,
            compress=3
        )


        joblib.dump(
            scaler,
            self.scaler_path,
            compress=3
        )


        joblib.dump(
            X_test,
            self.x_test_path,
            compress=3
        )


        joblib.dump(
            y_test,
            self.y_test_path,
            compress=3
        )


        print(
            "Model, scaler and test data saved successfully."
        )



    def run(self):


        X, y = self.load_data()


        X_train, X_test, y_train, y_test, scaler = self.preprocess(
            X,
            y
        )


        models = self.train_models(
            X_train,
            y_train
        )


        best_model = self.evaluate_models(
            models,
            X_test,
            y_test
        )


        self.save_files(
            best_model,
            scaler,
            X_test,
            y_test
        )



if __name__ == "__main__":


    trainer = ModelTrainer()

    trainer.run()