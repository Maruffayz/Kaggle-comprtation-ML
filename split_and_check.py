import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_clean.csv')
X = train.drop(columns=['Churn', 'id'])
y = train['Churn'].map({'No':0, 'Yes':1})

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print('Train/Test split done.')
print(f'X_train: {X_train.shape}, X_val: {X_val.shape}')
print(f'y_train: {y_train.shape}, y_val: {y_val.shape}')

print('\nOriginal target distribution:')
print(y.value_counts(normalize=True).rename('ratio'))

print('\nTrain target distribution:')
print(y_train.value_counts(normalize=True).rename('ratio'))

print('\nVal target distribution:')
print(y_val.value_counts(normalize=True).rename('ratio'))

orig = y.value_counts(normalize=True).iloc[0] / y.value_counts(normalize=True).iloc[1]
train_imb = y_train.value_counts(normalize=True).iloc[0] / y_train.value_counts(normalize=True).iloc[1]
val_imb = y_val.value_counts(normalize=True).iloc[0] / y_val.value_counts(normalize=True).iloc[1]

print(f'\nImbalance original: {orig:.4f}')
print(f'Imbalance train: {train_imb:.4f}')
print(f'Imbalance val: {val_imb:.4f}')

X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)

print('\nFiles saved: X_train.csv, X_val.csv, y_train.csv, y_val.csv')
