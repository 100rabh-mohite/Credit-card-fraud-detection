import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:

    def __init__(self):

        self.data_path = "data/creditcard_clean.csv"
        self.output_dir = "outputs"

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):

        print("=" * 60)
        print("Loading Clean Dataset...")
        print("=" * 60)

        df = pd.read_csv(self.data_path)

        return df

    def dataset_info(self, df):

        print("\nDataset Shape")
        print(df.shape)

        print("\nDataset Information")
        print(df.info())

        print("\nStatistical Summary")
        print(df.describe())

        print("\nFraud Distribution")
        print(df["Class"].value_counts())

    def class_distribution(self, df):

        plt.figure(figsize=(6,5))

        sns.countplot(x="Class", data=df)

        plt.title("Fraud vs Normal Transactions")

        plt.savefig(
            os.path.join(
                self.output_dir,
                "class_distribution.png"
            )
        )

        plt.close()

    def amount_distribution(self, df):

        plt.figure(figsize=(8,5))

        sns.histplot(
            df["Amount"],
            bins=50,
            kde=True
        )

        plt.title("Transaction Amount Distribution")

        plt.savefig(
            os.path.join(
                self.output_dir,
                "amount_distribution.png"
            )
        )

        plt.close()

    def amount_boxplot(self, df):

        plt.figure(figsize=(8,4))

        sns.boxplot(
            x=df["Amount"]
        )

        plt.title("Amount Boxplot")

        plt.savefig(
            os.path.join(
                self.output_dir,
                "amount_boxplot.png"
            )
        )

        plt.close()

    def correlation_heatmap(self, df):

        plt.figure(figsize=(14,10))

        sns.heatmap(
            df.corr(),
            cmap="coolwarm",
            center=0
        )

        plt.title("Correlation Heatmap")

        plt.savefig(
            os.path.join(
                self.output_dir,
                "correlation_heatmap.png"
            )
        )

        plt.close()

    def run(self):

        df = self.load_data()

        self.dataset_info(df)

        print("\nGenerating Graphs...")

        self.class_distribution(df)

        self.amount_distribution(df)

        self.amount_boxplot(df)

        self.correlation_heatmap(df)

        print("\nEDA Completed Successfully.")
        print(f"Graphs saved in '{self.output_dir}' folder.")


if __name__ == "__main__":

    eda = EDA()

    eda.run()