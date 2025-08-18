# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:27:55 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("airline.csv")   # Replace with actual filename

# ==============================
# 1. Outlier Treatment
# ==============================

# --- 1st Method: Calculate IQR ---
IQR = df['Balance'].quantile(0.75) - df['Balance'].quantile(0.25)
lower_limit = df['Balance'].quantile(0.25) - 1.5 * IQR
upper_limit = df['Balance'].quantile(0.75) + 1.5 * IQR

sns.boxplot(df['Balance'])

# --- 2nd Method: Trimming ---
outlier_df = np.where(df['Balance'] > upper_limit, True,
              np.where(df['Balance'] < lower_limit, True, False))
df_trimmed = df.loc[~outlier_df]

# --- 3rd Method: Replacement Technique ---
df_replaced = pd.DataFrame(
    np.where(df['Balance'] > upper_limit, upper_limit,
    np.where(df['Balance'] < lower_limit, lower_limit, df['Balance'])),
    columns=['Balance']
)
sns.boxplot(df_replaced['Balance'])

# --- 4th Method: Winsorizer ---
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Balance'])
df_winsor = winsor.fit_transform(df[['Balance']])
sns.boxplot(df_winsor['Balance'])

# ==============================
# 2. Zero Variance Check
# ==============================

# 1. Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# 2. Find variance of numeric columns
variances = numeric_df.var()

# 3. Identify zero variance columns
zero_var_cols = variances[variances == 0].index.tolist()

# 4. Drop zero variance columns
df_cleaned = df.drop(columns=zero_var_cols)
print("Zero Variance Columns:", zero_var_cols)

# ==============================
# 3. Imputation Techniques
# ==============================

# --- Mean Imputation (example: Balance) ---
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['Balance'] = mean_imputer.fit_transform(df[['Balance']])

# --- Median Imputation (example: Qual_miles) ---
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df['Qual_miles'] = median_imputer.fit_transform(df[['Qual_miles']])

# --- Mode Imputation (example: Award) ---
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['Award'] = mode_imputer.fit_transform(df[['Award']])

# Check final missing values
print(df.isnull().sum())
