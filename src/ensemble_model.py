import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

# Load cleaned training data
train = pd.read_csv('data/train_clean.csv')
X = train.drop(columns=['Churn', 'id'])
y = train['Churn'].map({'No':0, 'Yes':1}).astype(int)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# LightGBM model
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(30)])

# XGBoost model using native API for compatibility
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val = xgb.DMatrix(X_val, label=y_val)

xgb_params = {
    'objective': 'binary:logistic',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'seed': 42
}

xgb_bst = xgb.train(xgb_params, xgb_train, num_boost_round=300, evals=[(xgb_val, 'valid')], early_stopping_rounds=30, verbose_eval=False)

# Predictions on validation
y_val_lgb = lgb_model.predict_proba(X_val)[:,1]
best_iter = getattr(xgb_bst, 'best_iteration', None)
if best_iter is None:
    best_iter = getattr(xgb_bst, 'best_ntree_limit', None)

if best_iter is not None:
    y_val_xgb = xgb_bst.predict(xgb_val, iteration_range=(0, best_iter))
else:
    y_val_xgb = xgb_bst.predict(xgb_val)

# Simple average ensemble
ensemble_val = (y_val_lgb + y_val_xgb) / 2
ensemble_auc = roc_auc_score(y_val, ensemble_val)
print('LightGBM AUC:', roc_auc_score(y_val, y_val_lgb))
print('XGBoost AUC:', roc_auc_score(y_val, y_val_xgb))
print('Ensemble AUC:', ensemble_auc)

# Refit models on full train set (all data) for final predictions
lgb_model.fit(X, y)
xgb_full = xgb.DMatrix(X, label=y)
best_iter_full = getattr(xgb_bst, 'best_iteration', None)
if best_iter_full is None:
    best_iter_full = getattr(xgb_bst, 'best_ntree_limit', None)

n_rounds = best_iter_full if best_iter_full is not None else 300
xgb_bst_full = xgb.train(xgb_params, xgb_full, num_boost_round=n_rounds)

# predict test
X_test = pd.read_csv('data/test_clean.csv').drop(columns=['id'])

probs_lgb = lgb_model.predict_proba(X_test)[:,1]
probs_xgb = xgb_bst_full.predict(xgb.DMatrix(X_test), iteration_range=(0, n_rounds))

ensemble_probs = (probs_lgb + probs_xgb)/2

submission = pd.DataFrame({'id': pd.read_csv('data/test_clean.csv')['id'], 'Churn': ensemble_probs})
submission.to_csv('outputs/submission_ensemble.csv', index=False)
print('Saved outputs/submission_ensemble.csv')

# Save model comparison summary
with open('outputs/model_comparison.txt', 'a') as f:
    f.write(f'Ensemble AUC: {ensemble_auc:.6f}\n')
