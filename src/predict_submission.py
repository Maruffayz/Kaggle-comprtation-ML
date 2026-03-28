import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

print('Loading data...')
test_df = pd.read_csv('test_clean.csv')
X_test = test_df.drop(columns=['id'])

X_train = pd.read_csv('X_train.csv')
scaler = StandardScaler()
num_features = X_train.select_dtypes(include=['number']).columns
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print('Loading tuned model...')
model = xgb.XGBClassifier()
model.load_model('xgb_churn_tuned.model')

print('Predicting probabilities...')
probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'Churn': probs})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')

importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=['feature','importance']).sort_values(by='importance', ascending=False)
importance_df.to_csv('feature_importance.csv', index=False)
print('Saved feature_importance.csv')
print(importance_df.head(15))
print('Done')
