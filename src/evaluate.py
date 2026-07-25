import os
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


class ModelEvaluator:


    def __init__(self):

        self.model_path = "models/model.pkl"
        self.x_test_path = "models/X_test.pkl"
        self.y_test_path = "models/y_test.pkl"

        self.output_path = "outputs"

        os.makedirs(
            self.output_path,
            exist_ok=True
        )



    def load_files(self):

        print("Loading model and test data...")


        model = joblib.load(
            self.model_path
        )


        X_test = joblib.load(
            self.x_test_path
        )


        y_test = joblib.load(
            self.y_test_path
        )


        return model, X_test, y_test



    def evaluate_model(
            self,
            model,
            X_test,
            y_test
    ):


        print("\nEvaluating Model...\n")


        y_pred = model.predict(
            X_test
        )


        y_prob = model.predict_proba(
            X_test
        )[:,1]



        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred
        )


        recall = recall_score(
            y_test,
            y_pred
        )


        f1 = f1_score(
            y_test,
            y_pred
        )


        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )



        print("=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)


        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )



        report = classification_report(
            y_test,
            y_pred
        )


        print("\nClassification Report\n")

        print(report)



        # Save metrics

        with open(
            os.path.join(
                self.output_path,
                "metrics.txt"
            ),
            "w"
        ) as file:


            file.write(
                "MODEL PERFORMANCE\n"
            )

            file.write(
                "=" * 50 + "\n\n"
            )


            file.write(
                f"Accuracy  : {accuracy:.4f}\n"
            )

            file.write(
                f"Precision : {precision:.4f}\n"
            )

            file.write(
                f"Recall    : {recall:.4f}\n"
            )

            file.write(
                f"F1 Score  : {f1:.4f}\n"
            )

            file.write(
                f"ROC-AUC   : {roc_auc:.4f}\n\n"
            )


            file.write(
                report
            )



        # Confusion Matrix


        cm = confusion_matrix(
            y_test,
            y_pred
        )


        display = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )


        display.plot()


        plt.title(
            "Confusion Matrix"
        )


        plt.savefig(
            os.path.join(
                self.output_path,
                "confusion_matrix.png"
            ),
            bbox_inches="tight"
        )


        plt.close()



        # ROC Curve


        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )


        plt.title(
            "ROC Curve"
        )


        plt.savefig(
            os.path.join(
                self.output_path,
                "roc_curve.png"
            ),
            bbox_inches="tight"
        )


        plt.close()



        # Feature Importance


        if hasattr(
            model,
            "feature_importances_"
        ):


            importance = pd.DataFrame({

                "Feature_Index":
                range(
                    len(
                        model.feature_importances_
                    )
                ),


                "Importance":
                model.feature_importances_

            })


            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )



            importance.head(10).to_csv(
                os.path.join(
                    self.output_path,
                    "feature_importance.csv"
                ),
                index=False
            )



            plt.figure(
                figsize=(8,5)
            )


            plt.bar(
                importance.head(10)["Feature_Index"],
                importance.head(10)["Importance"]
            )


            plt.title(
                "Top 10 Feature Importance"
            )


            plt.xlabel(
                "Feature Index"
            )


            plt.ylabel(
                "Importance"
            )


            plt.tight_layout()



            plt.savefig(
                os.path.join(
                    self.output_path,
                    "feature_importance.png"
                ),
                bbox_inches="tight"
            )


            plt.close()



        plt.close("all")


        print(
            "\nEvaluation Completed Successfully."
        )


        print(
            "\nResults saved in 'outputs' folder."
        )




    def run(self):


        model, X_test, y_test = self.load_files()


        self.evaluate_model(
            model,
            X_test,
            y_test
        )

if __name__ == "__main__":


    evaluator = ModelEvaluator()

    evaluator.run()