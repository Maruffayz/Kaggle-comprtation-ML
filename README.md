# Kaggle Competition ML Pipeline

This repository contains a complete end-to-end pipeline for the Kaggle churn prediction comp: data cleaning, splitting, XGBoost training, hyperparameter tuning, and submission creation.

## Project Files
- `train.csv`, `test.csv`: original dataset files.
- `train_clean.csv`, `test_clean.csv`: cleaned datasets (after preprocessing).
- `data_cleaning.py`: cleaning rules, duplicate handling, outlier capping, label encoding, feature engineering.
- `split_and_check.py`: stratified train/test split and class imbalance validation.
- `prepare_xgboost.py`: baseline training + RandomizedSearchCV hyperparameter tuning + validation AUC tracking.
- `predict_submission.py`: build and save `submission.csv` and `feature_importance.csv` from tuned model.
- `xgb_churn.model`, `xgb_churn_tuned.model`: trained models.
- `submission.csv`: final test predictions (probability of churn, model output).

## Requirements
- Python 3.8+ (tested on 3.13)
- pandas
- numpy
- scikit-learn
- xgboost

Install requirements:

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

1. Clean data:

```bash
python data_cleaning.py
```

2. Perform train/test split and check imbalance:

```bash
python split_and_check.py
```

3. Train XGBoost + tune hyperparameters:

```bash
python prepare_xgboost.py
```

4. Create test predictions and importance scores:

```bash
python predict_submission.py
```

## Results
- Baseline XGBoost validation ROC-AUC: `0.91210` (30 rounds)
- Tuned XGBoost validation ROC-AUC: `0.91507`
- Best tune parameters found:
  - `subsample=0.6`
  - `n_estimators=400`
  - `min_child_weight=3`
  - `max_depth=6`
  - `learning_rate=0.15`
  - `gamma=0.2`
  - `colsample_bytree=0.7`
- Generated files:
  - `feature_importance.csv` (feature gain ranking)
  - `submission.csv` (ID + predicted churn probability)

## Notes
- Data is processed in-place with category label encoding and outlier capping.
- `scale_pos_weight` is computed from class imbalance.
- Hyperparameter search uses `RandomizedSearchCV` with design to avoid heavy, long runs in interactive session.


