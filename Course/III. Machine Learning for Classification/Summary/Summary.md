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
