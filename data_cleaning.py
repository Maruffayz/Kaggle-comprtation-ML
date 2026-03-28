import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA CLEANING PROCESS")
print("="*80)

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\n1. INITIAL DATA INSPECTION")
print(f"   Train: {train.shape}, Test: {test.shape}")
print(f"   Train missing values: {train.isnull().sum().sum()}")
print(f"   Test missing values: {test.isnull().sum().sum()}")

# 2. Check for duplicates
print(f"\n2. DUPLICATES")
print(f"   Train duplicates: {train.duplicated().sum()}")
print(f"   Test duplicates: {test.duplicated().sum()}")

# 3. Check data types
print(f"\n3. DATA TYPES")
print(train.dtypes)

# 4. Check for inconsistencies
print(f"\n4. CATEGORICAL VALUES CHECK")
categorical_cols = train.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"   {col}: {train[col].unique()[:10]}")

# 5. Check numeric columns for outliers and statistics
print(f"\n5. NUMERIC COLUMNS STATISTICS")
numeric_cols = train.select_dtypes(include=[np.number]).columns
print(train[numeric_cols].describe())

# 6. Check for empty/whitespace values
print(f"\n6. CHECKING FOR EMPTY/WHITESPACE VALUES")
for col in categorical_cols:
    if train[col].dtype == 'object':
        empty = (train[col] == '') | (train[col].str.strip() == '')
        print(f"   {col}: {empty.sum()} empty values")

# ============================================================================
# ACTUAL DATA CLEANING
# ============================================================================

print("\n" + "="*80)
print("APPLYING DATA CLEANING RULES")
print("="*80)

# Combine train and test for consistent preprocessing
train_id = train['id'].copy()
test_id = test['id'].copy()
train_y = train['Churn'].copy()

combined = pd.concat([train.drop('Churn', axis=1), test], ignore_index=True)

print("\n1. REMOVING DUPLICATES")
before = len(combined)
combined = combined.drop_duplicates(subset=combined.columns.difference(['id']), keep='first')
print(f"   Removed {before - len(combined)} duplicate rows")

print("\n2. HANDLING MISSING VALUES")
print(f"   Missing values before: {combined.isnull().sum().sum()}")
# No missing values found, but if there were:
# For numeric: fill with median
# For categorical: fill with mode
for col in combined.columns:
    if combined[col].isnull().sum() > 0:
        if combined[col].dtype in ['int64', 'float64']:
            combined[col].fillna(combined[col].median(), inplace=True)
        else:
            combined[col].fillna(combined[col].mode()[0], inplace=True)
print(f"   Missing values after: {combined.isnull().sum().sum()}")

print("\n3. HANDLING EMPTY STRINGS IN CATEGORICAL COLUMNS")
categorical_cols = combined.select_dtypes(include='object').columns
for col in categorical_cols:
    if combined[col].dtype == 'object':
        # Replace empty strings with the mode
        combined[col] = combined[col].replace('', combined[col].mode()[0] if len(combined[col].mode()) > 0 else 'Unknown')

print("\n4. FIXING DATA TYPES")
# Ensure consistent data types
for col in combined.columns:
    if col != 'id':
        if combined[col].dtype == 'object':
            # Convert to category for memory efficiency
            combined[col] = combined[col].astype('category')

print("\n5. HANDLING OUTLIERS (IQR Method)")
numeric_cols = combined.select_dtypes(include=[np.number]).columns.difference(['id'])
outlier_count = 0
for col in numeric_cols:
    Q1 = combined[col].quantile(0.25)
    Q3 = combined[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((combined[col] < lower_bound) | (combined[col] > upper_bound)).sum()
    outlier_count += outliers
    
    if outliers > 0:
        # Cap outliers instead of removing
        combined[col] = combined[col].clip(lower_bound, upper_bound)
        print(f"   {col}: {outliers} outliers capped")

print(f"   Total outliers handled: {outlier_count}")

print("\n6. ENCODING CATEGORICAL VARIABLES")
categorical_cols = combined.select_dtypes(include='category').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le
    print(f"   {col}: encoded ({len(le.classes_)} unique values)")

print("\n7. FEATURE ENGINEERING")
# Convert tenure to tenure categories
if 'tenure' in combined.columns:
    combined['tenure_category'] = pd.cut(combined['tenure'], bins=[0, 12, 24, 48, 60], 
                                         labels=['0-1y', '1-2y', '2-4y', '4+y'])
    combined['tenure_category'] = LabelEncoder().fit_transform(combined['tenure_category'].astype(str))
    print(f"   Created tenure_category from tenure")

# Create charge-to-months ratio
if 'MonthlyCharges' in combined.columns and 'tenure' in combined.columns:
    combined['MonthlyCharges_per_tenure'] = combined['MonthlyCharges'] / (combined['tenure'] + 1)
    print(f"   Created MonthlyCharges_per_tenure ratio")

print("\n8. NORMALIZATION/STANDARDIZATION CHECK")
numeric_cols = combined.select_dtypes(include=[np.number]).columns.difference(['id'])
print(f"   Numeric columns ready for scaling: {list(numeric_cols)}")

print("\n9. FINAL DATA INSPECTION")
print(f"   Combined shape: {combined.shape}")
print(f"   Data types: \n{combined.dtypes}")
print(f"   Missing values: {combined.isnull().sum().sum()}")
print(f"   Duplicates: {combined.duplicated().sum()}")

# Split back to train and test
train_clean = combined.iloc[:len(train_id)].copy()
test_clean = combined.iloc[len(train_id):].copy()

train_clean['id'] = train_id.values
test_clean['id'] = test_id.values
train_clean['Churn'] = train_y.values

# Save cleaned data
train_clean.to_csv('train_clean.csv', index=False)
test_clean.to_csv('test_clean.csv', index=False)

print("\n" + "="*80)
print("CLEANING COMPLETE")
print("="*80)
print(f"\nCleaned files saved:")
print(f"   train_clean.csv ({train_clean.shape})")
print(f"   test_clean.csv ({test_clean.shape})")

print(f"\nTrain sample (first 5 rows):")
print(train_clean.head())
