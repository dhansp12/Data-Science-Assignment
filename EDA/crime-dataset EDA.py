# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:30:55 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/dhana/Downloads/Crime-dataset.xlsx")

# Basic info
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nData Types:\n", df.dtypes)
print("\nSize of Dataset:", df.size)
print("\nMissing Values:\n", df.isnull().sum())

print("\nDescriptive Statistics:\n", df.describe())

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

# Plot distributions
for col in ['Murder', 'Assault', 'UrbanPop', 'Rape']:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Fourth Moment: Kurtosis
kur = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kur)

# Histograms of all numeric features
df.hist(figsize=(10,8), color='red', edgecolor='black')
plt.suptitle('Histogram of Crime Dataset Features')
plt.tight_layout()
plt.show()

# Boxplots (outlier detection)
for col in ['Murder', 'Assault', 'UrbanPop', 'Rape']:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Crime Features")
plt.show()

# Scatterplot Example
sns.scatterplot(data=df, x='Murder', y='Assault', hue='UrbanPop', size='Rape', sizes=(20,200))
plt.title("Murder vs Assault with UrbanPop & Rape Influence")
plt.show()

"""
Business / Analytical Insights:
1.Murder
High murder rates → unsafe areas, government may increase law enforcement.
Low murder rates → safer cities, good for investment and tourism.
2.Assault
High assault numbers → weak public safety, need more security measures.
Low assault → safer environment for people and businesses.
3.UrbanPop (Urban Population %)
Higher urban population → more facilities, but also more crime risk.
Lower urban population → rural areas, less crime but fewer opportunities.
4.Rape
High rape cases → urgent need for social awareness, strict law enforcement.
Low cases → safer community perception.
5.Name (State/Region name)
Helps to identify which state/region has higher or lower crime.
Useful for comparing crime patterns across locations.
"""
