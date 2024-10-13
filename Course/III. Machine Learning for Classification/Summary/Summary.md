# ML Zoomcamp 2024: Introduction to Machine Learning  

### Part 1

## Churn Prediction

### Introduction

The project aims to identify customers likely to churn (stop using a service) for a telecommunications company. Each customer is assigned a score representing their probability of churning. The goal is to use this data to send targeted promotions, such as discounts, to retain customers.

### What is Churn Prediction?

Churn prediction is a type of binary classification problem where the objective is to predict whether a customer will leave a service (churn) or stay. The model's output is a score between 0 and 1, indicating the likelihood of churning:

- **0**: Customer did not churn (negative example)
- **1**: Customer churned (positive example)

This scoring allows the company to proactively address customer dissatisfaction.

### The Approach

The approach to this problem involves the following key components:

- **Features (X)**: The dataset contains various customer information, such as demographics, payment details, service usage, and contract types.
- **Target Variable (y)**: The target variable indicates whether a customer has churned. Historical data is used to label customers who left the company as 1 and those who stayed as 0.

### Model Development

To develop the model:

1. **Data Collection**: Historical data from the Kaggle dataset titled "Telco Customer Churn – Focused Customer Retention Programs" is used.
2. **Labeling**: For the customers observed last month, those who left the service are labeled with a 1, while those who remained are labeled with a 0.
3. **Model Training**: The model learns from the historical data to identify patterns and relationships between customer characteristics and the likelihood of churn.

### Part 2

# Data Preparation Notes

## Session Overview
This session focused on data acquisition and preparation procedures.

### Commands, Functions, and Methods
- `!wget`: Linux shell command for downloading data.
- `pd.read_csv()`: Reads CSV files into a DataFrame.
- `df.head()`: Displays the first few rows of the DataFrame.
- `df.head().T`: Displays the transposed DataFrame.
- `df.columns`: Retrieves column names of the DataFrame.
- `df.columns.str.lower()`: Converts all column names to lowercase.
- `df.columns.str.replace(' ', '_')`: Replaces spaces in column names with underscores.
- `df.dtypes`: Retrieves data types of all series in the DataFrame.
- `df.index`: Retrieves indices of the DataFrame.
- `pd.to_numeric()`: Converts series values to numeric; the `errors='coerce'` argument allows conversion despite errors.
- `df.fillna()`: Replaces missing values (NAs) with specified values.
- `(df.x == "yes").astype(int)`: Converts 'yes-no' values in a series to numerical values.

## Data Preparation Topics Covered

1. Downloading the data
2. Reading the data
3. Standardizing column names and values
4. Verifying proper column reading
5. Checking if the churn variable requires preparation

### Downloading the Data
First, import necessary packages and download the CSV file using the `wget` command. In Jupyter Notebook, the `!` indicates a shell command, while the `$` symbol is used to reference variables within that command.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = "https://..."
!wget $data -O data-week-3.csv
```

To display all of them simultaneously, we can use the transpose function. This will switch the rows to become columns and the columns to become rows.

```python
df.head().T
```
