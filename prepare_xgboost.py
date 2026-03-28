import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load cleaned & split data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()

print('Shapes:', X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Optional: Feature scaling (XGBoost handles unscaled but scaling can help if large numeric range)
scaler = StandardScaler()
num_features = X_train.select_dtypes(include=[np.number]).columns
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_val[num_features] = scaler.transform(X_val[num_features])

# Class weights (inverse freq) for imbalance handling
class_counts = y_train.value_counts().to_dict()
total = len(y_train)
weights = y_train.map(lambda c: total / (len(class_counts) * class_counts[c]))

print('Class counts:', class_counts)
print('Sample weight example', weights.iloc[:5].to_list())

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
dval = xgb.DMatrix(X_val, label=y_val)

# Save DMatrix (optional) for faster reuse
xgb.DMatrix.save_binary(dtrain, 'dtrain.buffer')
xgb.DMatrix.save_binary(dval, 'dval.buffer')

print('DMatrix prepared and saved: dtrain.buffer, dval.buffer')

# Example params for binary classification
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': class_counts[0] / class_counts[1],
    'seed': 42,
}

print('XGBoost params prepared:', params)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Direct XGBoost train for baseline (short to avoid long runtime)
bst = xgb.train(params, dtrain, num_boost_round=30, evals=[(dtrain, 'train'), (dval, 'val')], early_stopping_rounds=10)
print('Baseline XGBoost Best AUC (30 rounds):', bst.best_score)

# Hyperparameter tuning with RandomizedSearchCV
print('\nStarting RandomizedSearchCV hyperparameter tuning...')

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=class_counts[0] / class_counts[1],
    random_state=42,
    n_jobs=-1,
)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'min_child_weight': [1, 3, 5, 7],
}

rs = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=3,  # quick completion for interactive step
    scoring='roc_auc',
    cv=2,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

rs.fit(X_train, y_train)

print('\nRandomizedSearchCV complete')
print('Best params:', rs.best_params_)
print('Best CV AUC:', rs.best_score_)

# Evaluate best model on validation set
y_val_pred = rs.best_estimator_.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print('Validation ROC AUC with tuned model:', val_auc)

# Save tuned model
rs.best_estimator_.save_model('xgb_churn_tuned.model')
print('Tuned model saved as xgb_churn_tuned.model')

