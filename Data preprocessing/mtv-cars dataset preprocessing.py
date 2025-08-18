# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:51:47 2025

@author: dhana
"""

# ==========================
# Data Preprocessing Code
# ==========================

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

# Load dataset
df = pd.read_excel("C:/Users/dhana/Downloads/mtvcars-dataset.xlsx")   # replace with your file

# ==========================
# 1. Outlier Treatment
# ==========================

# Example on 'mpg' column (can be repeated for others like hp, wt, disp, qsec)

# --- 1st Method: Calculate IQR ---
IQR = df['mpg'].quantile(0.75) - df['mpg'].quantile(0.25)
lower_limit = df['mpg'].quantile(0.25) - 1.5 * IQR
upper_limit = df['mpg'].quantile(0.75) + 1.5 * IQR

sns.boxplot(df['mpg'])

# --- 2nd Method: Trimming ---
outlier_df = np.where(df['mpg'] > upper_limit, True,
                      np.where(df['mpg'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

print("Before Trimming:", df.shape, "After Trimming:", df_trimmed.shape)

# --- 3rd Method: Replacement Technique ---
df_replaced = pd.DataFrame(
    np.where(df['mpg'] > upper_limit, upper_limit,
    np.where(df['mpg'] < lower_limit, lower_limit, df['mpg'])),
    columns=['mpg']
)
sns.boxplot(df_replaced['mpg'])

# --- 4th Method: Winsorizer ---
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['mpg'])
df_winsor = winsor.fit_transform(df[['mpg']])
sns.boxplot(df_winsor['mpg'])

# ==========================
# 2. Zero Variance Check
# ==========================

# 1. Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# 2. Find variance of numeric columns
variances = numeric_df.var()

# 3. Identify zero variance columns
zero_var_cols = variances[variances == 0].index.tolist()

# 4. Drop those columns
df_cleaned = df.drop(columns=zero_var_cols)
print("Zero Variance Columns:", zero_var_cols)

# ==========================
# 3. Imputation Techniques
# ==========================

# --- Mean Imputation (example: mpg) ---
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['mpg'] = mean_imputer.fit_transform(df[['mpg']])

# --- Median Imputation (example: hp) ---
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df['hp'] = median_imputer.fit_transform(df[['hp']])

# --- Mode Imputation (example: gear) ---
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['gear'] = mode_imputer.fit_transform(df[['gear']])

# ==========================
# Final Check
# ==========================
print("Missing Values After Imputation:\n", df.isnull().sum())
