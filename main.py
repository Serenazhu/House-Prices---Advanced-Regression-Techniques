import pandas as pd # for viewing/working with the data
from sklearn.linear_model import LinearRegression # for the machine learnig model
import numpy as np # also for data manipulation
import os # for working with files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load the datasets
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Identify columns to drop due to high percentage of missing values
isnull = train_df.isnull().sum()
total = train_df.shape[0]
null_percentage = isnull / total
columns_to_drop = null_percentage[null_percentage > 0.70].index

# Drop columns from both datasets
train_df.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=columns_to_drop, inplace=True)

# Identify numeric columns
numeric_cols = train_df.select_dtypes(include=['number']).columns

# Temporary setting the target to 0
test_df['SalePrice'] = 0

# Fill missing values
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())

# Identify categorical columns
string_cols = train_df.select_dtypes(include=['object']).columns

# Fill missing values in string columns
train_df[string_cols] = train_df[string_cols].fillna(train_df[string_cols].mode().iloc[0])
test_df[string_cols] = test_df[string_cols].fillna(test_df[string_cols].mode().iloc[0])

# Apply one-hot encoding
train_encoded = pd.get_dummies(train_df.drop(columns=['SalePrice']))
test_encoded = pd.get_dummies(test_df)

# Align the test_encoded columns to match train_encoded columns
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Prepare final datasets
X_train = train_encoded.to_numpy()
X_test = test_encoded.to_numpy()
y_train = train_df['SalePrice'].to_numpy()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_df['SalePrice'] = y_pred
test_df[['Id', 'SalePrice']].to_csv('linear_regression_prediction4.csv', index = False)
