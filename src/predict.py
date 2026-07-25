import joblib
import pandas as pd


class FraudPredictor:

    def __init__(self):

        self.model_path = "models/model.pkl"
        self.scaler_path = "models/scaler.pkl"

        self.model = joblib.load(
            self.model_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )


    def predict(self, transaction):

        data = pd.DataFrame(
            [transaction]
        )

        data_scaled = self.scaler.transform(
            data
        )

        prediction = self.model.predict(
            data_scaled
        )

        probability = self.model.predict_proba(
            data_scaled
        )[0][1]


        if prediction[0] == 1:

            result = "Fraud Transaction"

        else:

            result = "Normal Transaction"


        return result, probability



if __name__ == "__main__":


    predictor = FraudPredictor()


    # Example transaction
    # Must contain the same 30 features as dataset

    sample_transaction = {

        "Time": 406,
        "V1": -1.35,
        "V2": 0.42,
        "V3": 1.15,
        "V4": 0.23,
        "V5": -0.12,
        "V6": 0.45,
        "V7": 0.31,
        "V8": -0.22,
        "V9": 0.11,
        "V10": -0.55,
        "V11": 0.25,
        "V12": -0.33,
        "V13": 0.04,
        "V14": -0.76,
        "V15": 0.18,
        "V16": -0.44,
        "V17": 0.32,
        "V18": -0.15,
        "V19": 0.21,
        "V20": 0.08,
        "V21": -0.10,
        "V22": 0.05,
        "V23": 0.02,
        "V24": 0.11,
        "V25": -0.04,
        "V26": 0.07,
        "V27": 0.01,
        "V28": 0.03,
        "Amount": 50

    }


    result, probability = predictor.predict(
        sample_transaction
    )


    print(
        "Prediction:",
        result
    )


    print(
        "Fraud Probability:",
        round(probability,4)
    )