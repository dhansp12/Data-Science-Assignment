# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:34:44 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/dhana/Downloads/advertising.csv.xlsx")

print("Shape of dataset:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nDataset Size:", df.size)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target variable distribution
print("\nClicked on Ad counts:")
print(df['Clicked_on_Ad'].value_counts())

# Descriptive statistics
print("\nSummary Statistics:")
print(df.describe())

"""
Moment-based understanding:
if mean < median → left skewed
if mean > median → right skewed
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
median_values = df.median(numeric_only=True)

#Daily_Time_Spent_on_Site: Mean is slightly higher than median
#Age: Mean is slightly less than median

print("\nMean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)

# 2nd Moment (Variance, Standard Deviation)
variance = df.var(numeric_only=True)
std_dev = df.std(numeric_only=True)

print("\nVariance:\n", variance)
print("\nStandard Deviation:\n", std_dev)

# High variance in Daily_Internet_Usage
# Low variance in Age 

# 3rd Moment (Skewness) 
skewness = df.skew(numeric_only=True)
print("\nSkewness:\n", skewness)

# Visualization
sns.histplot(df['Daily_Time_Spent_on_Site'], kde=True, color='blue')
plt.title("Daily Time Spent on Site Distribution")
plt.show()

sns.histplot(df['Daily_Internet_Usage'], kde=True, color='green')
plt.title("Daily Internet Usage Distribution")
plt.show()
#Daily_Internet_Usage: Right-skewed
#Age: Slight left skewness

# 4th Moment (Kurtosis)
kurtosis = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kurtosis)

# High kurtosis in Age
# Low kurtosis in Daily time on site

# Histograms
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Scatterplots
sns.scatterplot(data=df, x='Daily_Time_Spent_on_Site', y='Daily_Internet_Usage', hue='Clicked_on_Ad')
plt.title("Internet Usage vs Time on Site by Ad Click")
plt.show()

'''
1.Daily_Time_Spent_on_Site – Shows how long users spend on the website; more time may increase ad click chances.
2.Age – Different age groups respond differently to ads; helps in customer segmentation.
3.Area income – Higher income areas may click ads for premium products, while lower income areas may prefer budget products.
4.Daily_Internet_Usage – Users with high internet usage are more likely to see and click ads.
5.Ad_Topic_Line – Type of ad content; some ad topics may attract more clicks than others.
6.City – Helps track which city gives more engagement; useful for local targeting.
7.Male – Gender of user; helps analyze ad performance across male vs female users.
8.Country – Shows country-level ad response; useful for global marketing strategy.
9.Timestamp – Tells what time of day ads get the most clicks (morning, afternoon, night).
10.Clicked_on_Ad – Main target variable; shows if the user clicked the ad or not.
'''
