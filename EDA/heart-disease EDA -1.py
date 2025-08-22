# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 18:29:35 2025
"""
@author: dhana
'''
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("C:/Users/dhana/Downloads/heart disease dataset.xlsx")

# Basic info
print("Shape of dataset:", df.shape)
print("\nInfo:")
print(df.info())
print("\nData Types:\n", df.dtypes)
print("\nDataset Size:", df.size)
print("\nMissing Values:\n", df.isnull().sum())

# Value counts for target (if 'target' column exists)
if 'target' in df.columns:
    print("\nTarget Value Counts:\n", df['target'].value_counts())

# Descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())


# 1st Moment (Mean)
mean_value = df.mean(numeric_only=True)
print("\nMean Values:\n", mean_value)

# 2nd Moment (Variance, Std Dev)
variance = df.var(numeric_only=True)
print("\nVariance:\n", variance)

std_dev = df.std(numeric_only=True)
print("\nStandard Deviation:\n", std_dev)

# 3rd Moment (Skewness)
skewness = df.skew(numeric_only=True)
print("\nSkewness:\n", skewness)

# Plot distributions
for col in df.select_dtypes(include='number').columns:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 4th Moment (Kurtosis)
kurt = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kurt)

# Histograms of all numeric columns
df.hist(figsize=(12,10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

#INFERENCE of this code
#Mean (1st Moment): Typical heart patients are middle-aged with moderately high cholesterol.
#Variance (2nd Moment): Cholesterol and heart rate vary widely, making them key for risk segmentation.
#Skewness (3rd Moment): Outliers exist (very high cholesterol, very low heart rate), needing special medical focus.
#Kurtosis (4th Moment): Cholesterol has frequent extreme cases → crucial for healthcare cost planning and preventive intervention.

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for numerical columns
sns.pairplot(df, diag_kind='kde')
plt.show()

# Scatterplot examples (if columns exist)
if 'age' in df.columns and 'chol' in df.columns:
    sns.scatterplot(data=df, x='age', y='chol', hue='target' if 'target' in df.columns else None)
    plt.title("Age vs Cholesterol")
    plt.show()

if 'thalach' in df.columns and 'age' in df.columns:
    sns.scatterplot(data=df, x='age', y='thalach', hue='target' if 'target' in df.columns else None)
    plt.title("Age vs Max Heart Rate Achieved")
    plt.show()
