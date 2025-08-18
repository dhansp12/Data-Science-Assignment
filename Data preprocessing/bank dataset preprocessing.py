# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:27:52 2025

@author: dhana
"""

# ==============================================
# PREPROCESSING PIPELINE
# ==============================================

import pandas as pd
import numpy as np
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("bank_data.csv")   # replace with your dataset file

# ==============================================
# 1. OUTLIER TREATMENT (example on 'balance')
# ==============================================

# --- Method 1: Calculate IQR ---
IQR = df['balance'].quantile(0.75) - df['balance'].quantile(0.25)
lower_limit = df['balance'].quantile(0.25) - 1.5 * IQR
upper_limit = df['balance'].quantile(0.75) + 1.5 * IQR

print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# --- Method 2: Trimming ---
outlier_df = np.where(df['balance'] > upper_limit, True,
                      np.where(df['balance'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

# --- Method 3: Replacement Technique ---
df_replaced = pd.DataFrame(
    np.where(df['balance'] > upper_limit, upper_limit,
             np.where(df['balance'] < lower_limit, lower_limit, df['balance'])),
    columns=['balance']
)

# --- Method 4: Winsorizer ---
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['balance'])
df_winsor = winsor.fit_transform(df[['balance']])

# ==============================================
# 2. ZERO VARIANCE HANDLING
# ==============================================

# 1. Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# 2. Find variance of numeric columns
variances = numeric_df.var()

# 3. Identify zero variance columns
zero_var_cols = variances[variances == 0].index.tolist()
print("Zero Variance Columns:", zero_var_cols)

# 4. Drop them
df_cleaned = df.drop(columns=zero_var_cols)

# ==============================================
# 3. IMPUTATION TECHNIQUES
# ==============================================

df2 = df.copy()

# --- Mean Imputation (for Age) ---
mean_imputer = SimpleImputer(strategy='mean')
df2['Age'] = mean_imputer.fit_transform(df2[['Age']])

# --- Median Imputation (for duration) ---
median_imputer = SimpleImputer(strategy='median')
df2['duration'] = median_imputer.fit_transform(df2[['duration']])

# --- Mode Imputation (for categorical variables: default, housing, loan, poutfailure) ---
mode_imputer = SimpleImputer(strategy='most_frequent')
df2['default'] = mode_imputer.fit_transform(df2[['default']])
df2['housing'] = mode_imputer.fit_transform(df2[['housing']])
df2['loan'] = mode_imputer.fit_transform(df2[['loan']])
df2['poutfailure'] = mode_imputer.fit_transform(df2[['poutfailure']])

# ==============================================
# Final Dataset Ready
# ==============================================
print(df2.info())
print(df2.isnull().sum())
