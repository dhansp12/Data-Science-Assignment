# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:18:26 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/dhana/Downloads/bank data1.xlsx")

print("Shape of dataset:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nData Types:\n", df.dtypes)
print("\nSize of Dataset:", df.size)
print("\nMissing Values:\n", df.isnull().sum())

print("\nDescriptive Statistics:\n", df.describe(include='all'))

'''
First Moment (Mean & Median) – central tendency.
Second Moment (Variance & Standard Deviation) – dispersion.
Third Moment (Skewness) – symmetry of distribution.
Fourth Moment (Kurtosis) – peakness of distribution.
Visualizations: Histograms, Scatterplots, Pairplot, Heatmap.
'''

# First Moment: Mean & Median
mean_value = df.mean(numeric_only=True)
median_value = df.median(numeric_only=True)

print("\nMean Values:\n", mean_value)
print("\nMedian Values:\n", median_value)

# Second Moment: Variance & Standard Deviation
vari = df.var(numeric_only=True)
std = df.std(numeric_only=True)

print("\nVariance:\n", vari)
print("\nStandard Deviation:\n", std)

# Third Moment: Skewness
skw = df.skew(numeric_only=True)
print("\nSkewness:\n", skw)

# Plot distribution of numeric features
for col in ['Age','balance','duration','campaign','pdays','previous']:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Fourth Moment: Kurtosis
kur = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kur)

# Histograms
df.hist(figsize=(12,10), color='red', edgecolor='black')
plt.suptitle('Histogram of Numerical Features')
plt.tight_layout()
plt.show()

# Boxplots for outliers
for col in ['Age','balance','duration','campaign','pdays','previous']:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Scatterplot example: Age vs Balance
sns.scatterplot(data=df, x='Age', y='balance', hue='housing')
plt.title("Age vs Balance by Housing Loan Status")
plt.show()

"""
Business Insights from EDA:
Age

Younger people may be more flexible in trying new financial products.

Older customers may prefer stable savings rather than risky investments.

Default

If default rate is high, customers are financially stressed → risky to give loans.

If very few defaults, bank can safely offer more credit.

Balance (Bank account balance)

Higher balance = better financial health, more chance to accept new schemes.

Negative/low balance = struggling customers, less likely to invest.

Housing (Housing loan status)

Customers with housing loans already have long-term commitments.

They may hesitate to take additional loans but can be targeted for insurance.

Loan (Personal loan status)

Customers with personal loans may be under debt pressure → less chance to accept new offers.

No-loan customers are more financially free and open to products.

Duration (Call duration in campaign)

Longer calls → higher chance of success (customer is interested).

Very short calls → customer is not engaged.

Campaign (Number of contacts in campaign)

Too many contacts can annoy the customer → negative outcome.

Few but effective contacts → better conversion.

Pdays (Days passed after previous campaign contact)

Small pdays (recent contact) = customer may remember → higher success.

Large pdays (long gap) = customer may not recall → lower success.

Previous (Number of contacts before current campaign)

If many previous contacts but no success → customer may not be interested.

If few contacts and positive response → good target group.

Poutcome/Failure (Outcome of previous campaign)

If previous outcome was success → high chance of success again.

If failure → customer might not be worth repeated calls.
