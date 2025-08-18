# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:12:53 2025

@author: dhana
"""

# PREPROCESSING PIPELINE

import pandas as pd
import numpy as np
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_excel("C:/Users/dhana/Downloads/advertising (1).csv")

# 1. OUTLIER TREATMENT

# Method 1: Calculate IQR 
IQR = df['Daily_Time_Spent_on_Site'].quantile(0.75) - df['Daily_Time_Spent_on_Site'].quantile(0.25)
lower_limit = df['Daily_Time_Spent_on_Site'].quantile(0.25) - 1.5 * IQR
upper_limit = df['Daily_Time_Spent_on_Site'].quantile(0.75) + 1.5 * IQR

print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Method 2: Trimming
outlier_df = np.where(df['Daily_Time_Spent_on_Site'] > upper_limit, True,
                      np.where(df['Daily_Time_Spent_on_Site'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

# Method 3: Replacement Technique
df_replaced = pd.DataFrame(
    np.where(df['Daily_Time_Spent_on_Site'] > upper_limit, upper_limit,
             np.where(df['Daily_Time_Spent_on_Site'] < lower_limit, lower_limit, df['Daily_Time_Spent_on_Site'])),
    columns=['Daily_Time_Spent_on_Site']
)

# Method 4: Winsorizer
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Daily_Time_Spent_on_Site'])

df_winsor = winsor.fit_transform(df[['Daily_Time_Spent_on_Site']])

# 2. ZERO VARIANCE HANDLING

# 1. Select numeric columns
numeric_df = df.select_dtypes(include='number')

# 2. Find variance of numeric columns
variances = numeric_df.var()

# 3. Identify zero variance columns
zero_var_cols = variances[variances == 0].index.tolist()
print("Zero Variance Columns:", zero_var_cols)

# 4. Drop them
df_cleaned = df.drop(columns=zero_var_cols)


# Example with missing values
df2 = df.copy()

# Mean Imputation (for Age) 
mean_imputer = SimpleImputer(strategy='mean')
df2['Age'] = mean_imputer.fit_transform(df2[['Age']])

# Median Imputation (for Daily_Internet_Usage) 
median_imputer = SimpleImputer(strategy='median')
df2['Daily_Internet_Usage'] = median_imputer.fit_transform(df2[['Daily_Internet_Usage']])

# Mode Imputation (for categorical columns like City, Country, Ad_Topic_Line)
mode_imputer = SimpleImputer(strategy='most_frequent')
df2['City'] = mode_imputer.fit_transform(df2[['City']])
df2['Country'] = mode_imputer.fit_transform(df2[['Country']])
df2['Ad_Topic_Line'] = mode_imputer.fit_transform(df2[['Ad_Topic_Line']])

# Final Dataset Ready
print(df2.info())
print(df2.isnull().sum())
