import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = 'Housing.csv'

df = pd.read_csv(dataset_path)

print("Dataset Information:")
print(df.info())

categorical = [col for col in df if df[col].dtype == 'object']
print("Categorical Attributes:", categorical)
numeric = [col for col in df if df[col].dtype == 'int64']
print("Numeric Attributes:", numeric)

print("Count of Missing Values:")
print(df.isnull().sum())

# Using Z Score to handle Outliers
z_scores = np.abs(zscore(df[numeric]))
df = df[(z_scores < 3).all(axis=1)]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Visualizations
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Boxplots for numerical columns
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Histograms for numerical columns
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Applying Z score normalization to all numeric attributes
df[numeric] = df[numeric].apply(zscore)

# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# One-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Linear Regression Model
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
r2_reg = r2_score(y_test, y_pred_reg)
print(f"\nLinear Regression Metrics: ")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_reg))}")
print(f"R^2: {r2_reg}")

# Training Random Forest Model
forest = RandomForestRegressor(n_estimators=500, random_state=0, criterion="friedman_mse", min_samples_split=5)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
r2_forest = r2_score(y_test, y_pred_forest)
print(f"\nRandom Forest Metrics: ")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_forest))}")
print(f"R^2: {r2_forest}")

# XGBoost Regressor Model
xgbr = xgb.XGBRegressor(n_estimators=1000, random_state=0, learning_rate=0.005)
xgbr.fit(X_train, y_train)
y_pred_xgbr = xgbr.predict(X_test)
r2_xgbr = r2_score(y_test, y_pred_xgbr)
print(f"\nXGBoost Metrics: ")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgbr))}")
print(f"R^2: {r2_xgbr}")

# Calculate weights based on R^2 values
total_r2 = r2_reg + r2_forest + r2_xgbr
weights = [r2_reg / total_r2, r2_forest / total_r2, r2_xgbr / total_r2]

# Ensemble predictions using weighted average
y_pred_ensemble = (weights[0] * y_pred_reg) + (weights[1] * y_pred_forest) + (weights[2] * y_pred_xgbr)
print(f"\nEnsemble Model Metrics: ")
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ensemble))}')
print(f'R^2: {r2_score(y_test, y_pred_ensemble)}')
