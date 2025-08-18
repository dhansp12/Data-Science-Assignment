# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:38:14 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/dhana/Downloads/mtvcars-dataset.xlsx")

# Basic dataset
print("Shape:", df.shape)
print("\nInfo:")
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


# First Moment (Central Tendency)
mean_value = df.mean(numeric_only=True)
median_value = df.median(numeric_only=True)

print("\nMean Values:\n", mean_value)
print("\nMedian Values:\n", median_value)

# Second Moment (Dispersion)
vari = df.var(numeric_only=True)
std = df.std(numeric_only=True)

print("\nVariance:\n", vari)
print("\nStandard Deviation:\n", std)

# Third Moment (Skewness)
skw = df.skew(numeric_only=True)
print("\nSkewness:\n", skw)

# Distribution plots for numeric columns
for col in df.columns:
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f"Distribution of {col}")
    plt.show()

# Fourth Moment (Kurtosis)
kur = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kur)

# ---------- Histograms ----------
df.hist(figsize=(12, 10), color='red', edgecolor='black')
plt.suptitle('Histogram of Numerical Features - mtcars')
plt.tight_layout()
plt.show()

# Boxplots for Outliers
for col in df.columns:
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()

# ---------- Correlation Heatmap ----------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of mtcars Dataset")
plt.show()

# ---------- Scatterplots ----------
sns.scatterplot(data=df, x='wt', y='mpg', hue='cyl', size='hp', sizes=(20, 200))
plt.title("Weight vs MPG with Cylinders & HP Influence")
plt.show()

sns.scatterplot(data=df, x='disp', y='hp', hue='gear')
plt.title("Displacement vs Horsepower by Gear")
plt.show()

'''
Murder

High murder rates → unsafe areas, government may increase law enforcement.

Low murder rates → safer cities, good for investment and tourism.

Assault

High assault numbers → weak public safety, need more security measures.

Low assault → safer environment for people and businesses.

UrbanPop (Urban Population %)

Higher urban population → more facilities, but also more crime risk.

Lower urban population → rural areas, less crime but fewer opportunities.

Rape

High rape cases → urgent need for social awareness, strict law enforcement.

Low cases → safer community perception.

Name (State/Region name)

Helps to identify which state/region has higher or lower crime.

Useful for comparing crime patterns across locations.
 Insights from EDA (mtcars dataset):
1. mpg, wt, cyl, disp, hp → Decide fuel efficiency vs power trade-off.

drat, qsec → Indicate speed/performance.
vs, am, gear, carb → Show design choices and driving style.
'''