import os
import pandas as pd


class DataPreprocessor:

    def __init__(self):

        self.input_path = "data/creditcard.csv"
        self.output_path = "data/creditcard_clean.csv"

    def load_data(self):

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        df = pd.read_csv(self.input_path)

        print(f"Dataset Shape : {df.shape}")

        return df

    def check_missing_values(self, df):

        print("\nChecking Missing Values...")

        missing = df.isnull().sum()

        print(missing[missing > 0])

    def remove_duplicates(self, df):

        print("\nChecking Duplicate Rows...")

        duplicate_rows = df.duplicated().sum()

        print(f"Duplicate Rows Found : {duplicate_rows}")

        if duplicate_rows > 0:

            print("\nRemoving Duplicate Rows...")

            df = df.drop_duplicates().reset_index(drop=True)

            print("Duplicates Removed Successfully.")

        else:

            print("No Duplicate Rows Found.")

        print(f"Dataset Shape After Cleaning : {df.shape}")

        return df

    def save_dataset(self, df):

        os.makedirs("data", exist_ok=True)

        df.to_csv(self.output_path, index=False)

        print(f"\nClean Dataset Saved Successfully.")
        print(f"Location : {self.output_path}")

    def run(self):

        df = self.load_data()

        self.check_missing_values(df)

        df = self.remove_duplicates(df)

        self.save_dataset(df)

        print("\nPreprocessing Completed Successfully.")


if __name__ == "__main__":

    processor = DataPreprocessor()

    processor.run()