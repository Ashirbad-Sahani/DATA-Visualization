#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Ashirbad Sahani
# Regd No:     2141019017
# Created:     20-07-2024
# Copyright:   (c) rudra 2024
# Licence:     <Ashirbad>
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

# Step 2: Find the shape
print("Shape of the dataset:", df.shape)

# Step 3: Check for null values
print("\nNull values in the dataset:\n", df.isnull().sum())

# Step 4: Find the data types
print("\nData types of each column:\n", df.dtypes)

# Step 5: Statistical Information
print("\nStatistical information:\n", df.describe())

# Step 6: Convert categorical columns to numerical columns
# In the Iris dataset, the 'species' column is categorical
df['species'] = df['species'].astype('category').cat.codes

print("\nData after converting categorical columns to numerical:\n", df.head())

# Step 7: Correlation matrix
print("\nCorrelation matrix:\n", df.corr())

# Step 8: Data Visualization using matplotlib

# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Histogram for each feature
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Histogram for Each Feature')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Boxplot for Each Feature')
plt.show()