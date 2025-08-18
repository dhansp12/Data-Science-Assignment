# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:21:08 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("C:/Users/dhana/Downloads/Company data.xlsx")

# Basic info
print(df.shape)
print(df.info())
print(df.dtypes)
print("Dataset size:", df.size)
print("Categorical feature counts:")
print(df['ShelveLoc'].value_counts())  

# Descriptive statistics
print(df.describe())

"""
Moment-based understanding:
if mean < median  → left skewed
if mean > median  → right skewed
"""
'''
First Moment (Mean & Median) – central tendency.
Second Moment (Variance & Standard Deviation) – dispersion.
Third Moment (Skewness) – symmetry of distribution.
Fourth Moment (Kurtosis) – peakness of distribution.
Visualizations: Histograms, Scatterplots, Pairplot, Heatmap.
moment (Kurtosis) → Peak & tails
'''

# 1st Moment (Mean, Median)
mean_values = df.mean(numeric_only=True)
print("Means:\n", mean_values)

median_values = df.median(numeric_only=True)
print("Medians:\n", median_values)

# Business decision example:
# If Price mean < median → more stores have higher prices
# If Sales mean > median → sales is right-skewed

# 2nd Moment (Variance, Standard Deviation)
variance = df.var(numeric_only=True)
print("Variance:\n", variance)

std_dev = df.std(numeric_only=True)
print("Standard Deviation:\n", std_dev)

# High variance in Advertising 
# Low variance in Price

# 3rd Moment (Skewness)
skewness = df.skew(numeric_only=True)
print("Skewness:\n", skewness)

# Plot distributions
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

sns.histplot(df['Advertising'], kde=True)
plt.title("Advertising Distribution")
plt.show()

# Business decision:
# Strong right skewness in Advertising -> low sales
# Sales slightly skewed → high sales

# 4th Moment (Kurtosis)
kurtosis = df.kurtosis(numeric_only=True)
print("Kurtosis:\n", kurtosis)

# Visualization
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Scatterplot example: Price vs Sales
sns.scatterplot(data=df, x='Price', y='Sales', hue='ShelveLoc')
plt.title("Price vs Sales by Shelf Location")
plt.show()


'''
1.Sales – Shows how much product is sold; main measure of business performance.
2.CompPrice – Competitor’s price; helps compare pricing strategy with competitors.
3.Income – Average income of customers; higher income may lead to higher sales.
4.Advertising – Money spent on ads; shows how promotion affects sales.
5.Population – Size of the market area; bigger population means more potential buyers.
6.Price – Selling price of the product; directly affects demand and sales.
7.ShelveLoc – Product shelf location (Good/Medium/Bad); better placement increases sales.
8.Age – Age of target customers; helps understand customer buying behavior.
9.Education – Education level of customers; may influence product choice and spending.
10.Urban – Whether customer lives in urban area; urban customers may buy differently than rural ones.
11.US – Indicates if the customer is from the US; useful for regional sales comparison.
'''
