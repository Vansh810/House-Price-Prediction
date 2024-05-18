# House-Price-Prediction
This repository contains a project for predicting house prices using machine learning techniques. The project includes data preprocessing, exploratory data analysis (EDA), and implementation of multiple regression models to provide accurate predictions.

# Dataset Used
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

# Description
1. *Data Preprocessing*: 
   - Reads the dataset from the provided CSV file.
   - Handles missing values.
   - Identifies categorical and numeric attributes.
   - Removes outliers using Z-score method.
   - Removes duplicate records.
   - Normalizes numeric attributes using Z-score normalization.

2. *Visualization*:
   - Displays correlation heatmap.
   - Plots pairplots for numeric attributes.
   - Generates boxplots and histograms for numeric attributes.

3. *Model Training*:
   - Splits the data into training and testing sets.
   - One-hot encodes categorical features.
   - Trains Linear Regression, Random Forest Regression, and XGBoost Regression models.
   - Evaluates models using Root Mean Squared Error (RMSE) and R-squared (R^2) metrics.

4. *Ensemble Model*:
   - Calculates weights based on R^2 values of individual models.
   - Combines predictions of individual models using a weighted average.