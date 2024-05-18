import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

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

# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# One-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
