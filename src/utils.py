import os
import joblib


class ModelUtils:
    """
    Utility class for saving and loading models.
    """

    @staticmethod
    def save_model(model, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(model, file_path)

        print(f"Model saved successfully: {file_path}")


    @staticmethod
    def load_model(file_path):

        model = joblib.load(file_path)

        print(f"Model loaded successfully: {file_path}")

        return model


class ScalerUtils:
    """
    Utility class for saving and loading scaler.
    """

    @staticmethod
    def save_scaler(scaler, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(scaler, file_path)

        print(f"Scaler saved successfully: {file_path}")


    @staticmethod
    def load_scaler(file_path):

        scaler = joblib.load(file_path)

        print(f"Scaler loaded successfully: {file_path}")

        return scaler