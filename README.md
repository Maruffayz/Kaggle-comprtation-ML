# Kaggle Competition ML Pipeline

This repository implements a modular, architecture-first data science pipeline for the churn prediction task.

## Architecture

```
.
├── data/                           # Raw and preprocessed datasets
│   ├── train.csv
│   ├── test.csv
│   ├── train_clean.csv
│   ├── test_clean.csv
│   ├── sample_submission.csv
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   ├── submission.csv
│   ├── feature_importance.csv
│   ├── dtrain.buffer
│   ├── dval.buffer
│   └── ...
├── src/                            # Scripts and pipeline components
│   ├── data_cleaning.py            # Cleaning, cat encoding, outlier capping, feature engineering
│   ├── split_and_check.py          # Stratified split + target imbalance check
│   ├── prepare_xgboost.py          # XGBoost base training + RandomizedSearchCV tuning + validation
│   ├── predict_submission.py       # Final test predictions + feature importance
│   └── README.md                   # Documentation and results
├── models/                         # Model artifacts
│   ├── xgb_churn.model
│   ├── xgb_churn_tuned.model
│   └── ...
├── requirements.txt                # Dependencies
└── README.md                       # This architecture summary
```

## Pipeline stages
1. **Data cleaning & feature engineering** (`data_cleaning.py`)
   - deduplicate, convert categories to labels
   - outlier capping (IQR)
   - `tenure_category`, `MonthlyCharges_per_tenure`

2. **Train/val split & imbalance check** (`split_and_check.py`)
   - stratified split 75/25
   - class ratio checks on train/val

3. **Model training + tuning** (`prepare_xgboost.py`)
   - baseline XGBoost (AUC tracking)
   - RandomizedSearchCV hyperparameter tuning
   - tuned model evaluation on validation set

4. **Final prediction and output** (`predict_submission.py`)
   - test prediction (probabilities)
   - feature importance CSV

## Results
- Baseline validation ROC-AUC: `0.91210`
- Tuned validation ROC-AUC: `0.91507`
- Best tuned params:
  - `subsample=0.6`, `n_estimators=400`, `min_child_weight=3`,
  - `max_depth=6`, `learning_rate=0.15`, `gamma=0.2`, `colsample_bytree=0.7`

## Requirement
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```


