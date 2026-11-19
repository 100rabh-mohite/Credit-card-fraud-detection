# Credit Card Fraud Detection

An end-to-end machine-learning portfolio project for identifying potentially fraudulent credit-card transactions in a highly imbalanced dataset.

**Author:** Saurabh Mohite  
**Language:** Python  
**Application:** Streamlit

> **Important:** This is an educational demonstration, not a production fraud-detection or payment-decision system.

## Project highlights

- Exploratory analysis and preparation of transaction data
- Four classification models: Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Stratified train / validation / test split
- Feature scaling fitted on training data only
- SMOTE applied to training data only
- Validation-based model selection and decision-threshold tuning
- Holdout evaluation using Precision, Recall, F1, ROC-AUC, and PR-AUC
- Confusion matrix, ROC curve, Precision-Recall curve, and feature-importance outputs
- Streamlit interface for single-transaction demonstration

## Dataset

This project is designed for the public **Credit Card Fraud Detection** dataset available on Kaggle.

- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Original dataset: 284,807 transactions and 31 columns
- Target: `Class`
  - `0`: normal transaction
  - `1`: fraudulent transaction
- Features: `Time`, `V1`–`V28`, and `Amount`

The original V1–V28 columns are anonymized PCA-transformed features. They are not ordinary customer or merchant attributes.

### Prepare the input file

The training script expects this file:

```
data/creditcard_clean.csv
```

Download the Kaggle dataset, inspect it, and prepare the input CSV with the 30 numeric feature columns and the `Class` target column. Keep the original column names and do not include an index column. The training script validates the target, missing values, and numeric feature types before training.

The dataset is intentionally not committed to this repository. Keep it in the local `data/` directory.

## Repository structure

```
Credit-card-fraud-detection/
├── data/
│   └── creditcard_clean.csv       # local dataset; not committed
├── models/                        # generated model artifacts; not committed
├── outputs/                       # generated evaluation reports
├── src/
│   ├── train.py                   # split, preprocess, train and select
│   ├── evaluate.py                # final holdout evaluation
│   └── predict.py                 # validated single-row prediction
├── app.py                         # Streamlit application
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data split

The data is split with stratification into:
- 60% training data
- 20% validation data
- 20% final test data

The final test set is held back until the model and decision threshold have been selected.

### 2. Preprocessing and imbalance handling

- Fit `StandardScaler` on the training partition only.
- Transform validation and test data using that fitted scaler.
- Apply SMOTE only to the scaled training partition.
- Never resample the validation or test partitions.

This prevents information from the validation and test data leaking into model training.

### 3. Model selection

The four models are trained on the resampled training data. The best model and its decision threshold are selected using validation F1-score. The test set is not used for model selection.

### 4. Final evaluation

Run the evaluator once after training. It reports:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)

For fraud detection, do not judge a model by accuracy alone. Precision and recall help explain false alerts and missed fraud cases.

## Setup

Python 3.10–3.12 is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the prepared CSV at `data/creditcard_clean.csv`.

## Run the project

From the repository root:

**1. Train and select the model**

```bash
python src/train.py
```

**2. Evaluate on the untouched test set**

```bash
python src/evaluate.py
```

**3. Run a sample prediction**

```bash
python src/predict.py
```

**4. Launch the web application**

```bash
streamlit run app.py
```

Training generates model artifacts in `models/`. Evaluation reports and plots are written to `outputs/`. These generated files are local and should be regenerated after changing the training code or dataset.

## Outputs

After successful training and evaluation, expect:

- `models/model.pkl` — selected classifier
- `models/scaler.pkl` — fitted training scaler
- `models/threshold.pkl` — validation-selected decision threshold
- `models/feature_names.pkl` — feature order used by the model
- `models/model_results.csv` — validation model comparison
- `outputs/test_metrics.csv` and `outputs/metrics.txt` — holdout metrics
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/precision_recall_curve.png`
- `outputs/feature_importance.csv` and `outputs/feature_importance.png` — for models that expose feature importances

## Limitations and responsible use

- The anonymized PCA features make manual transaction entry in the demo less intuitive; use compatible feature values from the dataset for meaningful demonstrations.
- A threshold optimized for F1 is not automatically the correct business threshold. Real deployments require cost-based threshold selection and monitoring.
- SMOTE and offline evaluation do not address every real-world issue, such as changing fraud patterns, delayed labels, or data drift.
- Do not use this application to make real financial decisions.

## Future improvements

- Add a reproducible data-preparation and exploratory-analysis notebook
- Add automated tests and continuous integration
- Track experiments and model versions
- Add cost-sensitive threshold analysis and drift monitoring
- Deploy behind authentication with monitoring and audit logging

## License and dataset attribution

Check the Kaggle dataset page and its stated terms before redistributing the data. This repository does not include the dataset.
