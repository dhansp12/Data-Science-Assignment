# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:43:30 2025

@author: dhana
"""

# ==========================
# Preprocessing Code
# ==========================

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

# Load dataset
df = pd.read_csv("heart.csv")   # replace with your filename

# ==========================
# 1. Outlier Treatment
# ==========================

# Example: Apply on 'age' column first (can repeat for other numeric features)

# --- 1st Method: Calculate IQR ---
IQR = df['age'].quantile(0.75) - df['age'].quantile(0.25)
lower_limit = df['age'].quantile(0.25) - 1.5 * IQR
upper_limit = df['age'].quantile(0.75) + 1.5 * IQR

sns.boxplot(df['age'])

# --- 2nd Method: Trimming ---
outlier_df = np.where(df['age'] > upper_limit, True,
                      np.where(df['age'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

print("Before Trimming:", df.shape, "After Trimming:", df_trimmed.shape)

# --- 3rd Method: Replacement Technique ---
df_replaced = pd.DataFrame(
    np.where(df['age'] > upper_limit, upper_limit,
    np.where(df['age'] < lower_limit, lower_limit, df['age'])),
    columns=['age']
)
sns.boxplot(df_replaced['age'])

# --- 4th Method: Winsorizer ---
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['age'])
df_winsor = winsor.fit_transform(df[['age']])
sns.boxplot(df_winsor['age'])

# ==========================
# 2. Zero Variance Check
# ==========================

# 1. Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# 2. Find variance of numeric columns
variances = numeric_df.var()

# 3. Identify zero variance columns
zero_var_cols = variances[variances == 0].index.tolist()

# 4. Drop those columns from the original DataFrame
df_cleaned = df.drop(columns=zero_var_cols)
print("Zero Variance Columns:", zero_var_cols)

# ==========================
# 3. Imputation Techniques
# ==========================

# --- Mean Imputation (example: trestbps) ---
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['trestbps'] = mean_imputer.fit_transform(df[['trestbps']])

# --- Median Imputation (example: chol) ---
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df['chol'] = median_imputer.fit_transform(df[['chol']])

# --- Mode Imputation (categorical features: sex, cp, thal) ---
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['sex'] = mode_imputer.fit_transform(df[['sex']])
df['cp'] = mode_imputer.fit_transform(df[['cp']])
df['thal'] = mode_imputer.fit_transform(df[['thal']])

# ==========================
# Final Check
# ==========================
print("Missing Values After Imputation:\n", df.isnull().sum())
