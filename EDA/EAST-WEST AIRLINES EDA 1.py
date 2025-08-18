# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:48:02 2025

@author: dhana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset

df = pd.read_excel("C:/Users/dhana/Downloads/East-West airlines 1.xlsx")
# Basic info
print(df.shape)
print(df.info())
print(df.dtypes)
print("Total size:", df.size)

# Check categorical target variable
print(df['Award?'].value_counts())

# Summary statistics
print(df.describe())

'''
First Moment (Mean & Median) – central tendency.
Second Moment (Variance & Standard Deviation) – dispersion.
Third Moment (Skewness) – symmetry of distribution.
Fourth Moment (Kurtosis) – peakness of distribution.
Visualizations: Histograms, Scatterplots, Pairplot, Heatmap.
moment (Kurtosis) → Peak & tails
'''


# First Moment – Central Tendency
mean_value = df.mean(numeric_only=True)
median_value = df.median(numeric_only=True)
print("\nMean:\n", mean_value)
print("\nMedian:\n", median_value)

# Business Insight Example:
# If mean > median → Right skew → high values
# If mean < median → Left skew → low values

# Second Moment – Dispersion
variance_value = df.var(numeric_only=True)
std_value = df.std(numeric_only=True)
print("\nVariance:\n", variance_value)
print("\nStandard Deviation:\n", std_value)

# Third Moment – Skewness
skew_value = df.skew(numeric_only=True)
print("\nSkewness:\n", skew_value)

# Fourth Moment – Kurtosis
kurtosis_value = df.kurtosis(numeric_only=True)
print("\nKurtosis:\n", kurtosis_value)

# Visualizations
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

# Scatter Example
sns.scatterplot(data=df, x="Balance", y="Bonus_miles", hue="Award?")
plt.title("Balance vs Bonus Miles by Award Status")
plt.show()

'''
1.ID – Unique number to identify each customer.
2.Balance – Shows total miles a customer has; higher balance means loyal or frequent flyer.
3.Qual_miles – Miles flown on qualifying flights; helps check loyalty program engagement.
4.cc1_miles, cc2_miles, cc3_miles – Miles earned through different 
5.credit cards; shows which card is used most by customers.
6.Bonus_miles – Extra miles earned; helps track promotional impact.
7.Bonus_trans – Number of bonus transactions; shows how often customers use offers.
8.Flight_miles_12mo – Miles flown in the last 12 months; indicates recent travel activity.
9.Flight_trans_12 – Number of flights taken in last 12 months; useful to find frequent flyers.
10.Days_since_enroll – How long the customer has been in the loyalty program; older members are more loyal.
11.Award – Whether the customer redeemed an award ticket; shows engagement with the program.
'''

