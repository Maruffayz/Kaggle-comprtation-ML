import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load cleaned data
train = pd.read_csv('data/train_clean.csv')

X = train.drop(columns=['Churn', 'id'])
y = train['Churn'].map({'No':0, 'Yes':1}).astype(int)

# Train/val split from clean data (again) with same stratify
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Build model
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=30)]
)

val_preds = model.predict_proba(X_val)[:,1]
val_auc = roc_auc_score(y_val, val_preds)
print('LightGBM validation AUC:', val_auc)

# Compare with xgboost by loading one trained model and predicting
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgb_churn_tuned.model')

xgb_preds = xgb_model.predict_proba(X_val)[:,1]
xgb_auc = roc_auc_score(y_val, xgb_preds)
print('XGBoost tuned validation AUC:', xgb_auc)

# Save comparison results
with open('outputs/model_comparison.txt', 'w') as f:
    f.write(f'LightGBM AUC: {val_auc:.6f}\n')
    f.write(f'XGBoost tuned AUC: {xgb_auc:.6f}\n')

print('Saved outputs/model_comparison.txt')
