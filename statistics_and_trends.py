"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as snsanalysis

data = pd.read_csv("data.csv")

data.head()

data.shape

data.isnull().sum()

data.describe()

data=data.dropna()

#Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#Histogram
col = "Credit_Limit" 

if col in data.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(data[col], bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
else:
    print(f"Column '{col}' not found in dataset.")



#Boxplot
if col in data.columns:
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=data[col], color="lightgreen")
    plt.title(f"Boxplot of {col}")
    plt.show()


#Scatter plot
if {"Total_Trans_Ct", "Total_Trans_Amt"}.issubset(data.columns):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=data, x="Total_Trans_Ct", y="Total_Trans_Amt", hue="Attrition_Flag")
    plt.title("Total Transactions Count vs Amount")
    plt.show()


#Basic Statistical Analysis
col = "Credit_Limit"  
if col in data.columns:
    x = data[col]
    mean = x.mean()
    std = x.std()
    skew = ss.skew(x)
    kurt = ss.kurtosis(x)

    print(f"\nFor column '{col}':")
    print(f"Mean = {mean:.2f}")
    print(f"Standard Deviation = {std:.2f}")
    print(f"Skewness = {skew:.2f}")
    print(f"Excess Kurtosis = {kurt:.2f}")

    if skew > 0.5:
        skew_text = "right-skewed"
    elif skew < -0.5:
        skew_text = "left-skewed"
    else:
        skew_text = "approximately symmetric"

    if kurt > 0.5:
        kurt_text = "leptokurtic"
    elif kurt < -0.5:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"

    print(f"The data is {skew_text} and {kurt_text}.")


#Separate Numerical & Categorical Columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

print("\nNumerical Columns:\n", numerical_cols.tolist())
print("\nCategorical Columns:\n", categorical_cols.tolist())

#Numerical Analysis
for col in numerical_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(data[col], bins=30, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


#Numerical Analysis
for col in numerical_cols:
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=data[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()

#Pairwise Relationships
key_cols = ['Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio']
existing_cols = [c for c in key_cols if c in data.columns]
if len(existing_cols) >= 2:
    sns.pairplot(data[existing_cols])
    plt.suptitle("Pairwise Relationships (Numerical Features)", y=1.02)
    plt.show()

#Value Counts for each Categorical Column
for col in categorical_cols:
    print(f"\nValue Counts for {col}:")
    print(data[col].value_counts(), "\n")

#Count Plots for each Categorical Feature
for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=data, x=col, palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=30)
    plt.show()

#Numerical vs Categorical Comparison
if 'Attrition_Flag' in data.columns and 'Credit_Limit' in data.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, x='Attrition_Flag', y='Credit_Limit', palette='coolwarm')
    plt.title("Credit Limit Distribution by Attrition_Flag")
    plt.show()

#Average Transcatiopn Amount by Card Category
if 'Card_Category' in data.columns and 'Total_Trans_Amt' in data.columns:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=data, x='Card_Category', y='Total_Trans_Amt', estimator='mean', ci=None, palette='mako')
    plt.title("Average Transaction Amount by Card Category")
    plt.xticks(rotation=30)
    plt.show()





