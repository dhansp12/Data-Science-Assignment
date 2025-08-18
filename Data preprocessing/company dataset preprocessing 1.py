# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:16:41 2025

@author: dhana
"""
# PREPROCESSING PIPELINE

import pandas as pd
import numpy as np
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_excel("C:/Users/dhana/Downloads/Company data.xlsx")

# 1. OUTLIER TREATMENT

# --- Method 1: Calculate IQR (example: on Sales) ---
IQR = df['Sales'].quantile(0.75) - df['Sales'].quantile(0.25)
lower_limit = df['Sales'].quantile(0.25) - 1.5 * IQR
upper_limit = df['Sales'].quantile(0.75) + 1.5 * IQR

print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# --- Method 2: Trimming ---
outlier_df = np.where(df['Sales'] > upper_limit, True,
                      np.where(df['Sales'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

# --- Method 3: Replacement Technique ---
df_replaced = pd.DataFrame(
    np.where(df['Sales'] > upper_limit, upper_limit,
             np.where(df['Sales'] < lower_limit, lower_limit, df['Sales'])),
    columns=['Sales']
)

# --- Method 4: Winsorizer ---
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Sales'])
df_winsor = winsor.fit_transform(df[['Sales']])

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

# Example dataset with missing values
df2 = df.copy()

# --- Mean Imputation (for Income) ---
mean_imputer = SimpleImputer(strategy='mean')
df2['Income'] = mean_imputer.fit_transform(df2[['Income']])

# --- Median Imputation (for Age) ---
median_imputer = SimpleImputer(strategy='median')
df2['Age'] = median_imputer.fit_transform(df2[['Age']])

# --- Mode Imputation (for categorical columns like ShelveLoc, Urban, US) ---
mode_imputer = SimpleImputer(strategy='most_frequent')
df2['ShelveLoc'] = mode_imputer.fit_transform(df2[['ShelveLoc']])
df2['Urban'] = mode_imputer.fit_transform(df2[['Urban']])
df2['US'] = mode_imputer.fit_transform(df2[['US']])

# ==============================================
# Final Dataset Ready
# ==============================================
print(df2.info())
print(df2.isnull().sum())
