# ML Zoomcamp 2024: Evaluation metrics for classification

# Churn Prediction Model Using Logistic Regression

## Overview
This script covers the essential steps for building a churn prediction model using Logistic Regression. We will:
1. Import necessary libraries.
2. Prepare the data by cleaning and transforming it.
3. Split the dataset into training, validation, and test sets.
4. Train a Logistic Regression model on the training data.
5. Validate the model on the validation data and evaluate its performance.

---

### 1. Necessary Imports

We start by importing the required libraries: Pandas for data manipulation, NumPy for numerical operations, Matplotlib for visualization, and several modules from Scikit-Learn for machine learning tasks.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
```

2. Data Preparation
Load the dataset and standardize column names by converting them to lowercase and replacing spaces with underscores.
Identify categorical columns and ensure that the 'totalcharges' column is correctly converted to a numerical format.
Convert the 'churn' column into a binary format, where 'yes' becomes 1 and 'no' becomes 0.


```python
df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype(int)
```
3. Data Splitting
We split the dataset into:

60% for training,
20% for validation, and
20% for testing.
The indices are reset to ensure continuous indexing, and the 'churn' column is separated as the target variable.

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```
4. Feature Preparation
We define two lists: one for numerical features and one for categorical features.

```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
```

5. Vectorization and Model Training
We use the DictVectorizer to transform the categorical and numerical columns into vectors, and then train the Logistic Regression model.
```python
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)
```

6. Model Validation
We transform the validation dataset similarly, predict churn probabilities, and evaluate the accuracy of the model.
```python
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
accuracy = (y_val == churn_decision).mean()

# Output the accuracy
accuracy
```

# Model Accuracy and Evaluation

## Accuracy and Dummy Model

In the previous analysis, our model achieved **80% accuracy** on the validation data, but we need to evaluate whether this is a good result.

**Accuracy** measures the proportion of correct predictions made by the model. In this case, a prediction was considered correct if a customer's predicted value was above the 0.5 threshold, meaning they were classified as "churn." Otherwise, they were classified as "non-churn."

Out of **1409 customers** in the validation dataset, the model correctly predicted the churn status for **1132 customers**, resulting in an accuracy of **80%**:
```python
len(y_val)  # Output: 1409
(y_val == churn_decision).sum()  # Output: 1132
1132 / 1409  # Output: 0.8034
(y_val == churn_decision).mean()  # Output: 0.8034
```

Evaluating Model on Different Thresholds
We can test if 0.5 is the best threshold for our model by experimenting with various threshold values. We can generate a range of values using NumPy's linspace function and evaluate the model at each threshold to find the one that maximizes accuracy.

```python
import numpy as np

thresholds = np.linspace(0, 1, 21)  # Generate thresholds from 0 to 1
scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```
```python
0.00 0.274
0.05 0.509
0.10 0.591
0.15 0.666
0.20 0.710
0.25 0.739
0.30 0.760
0.35 0.772
0.40 0.785
0.45 0.793
0.50 0.803
0.55 0.801
0.60 0.795
0.65 0.786
0.70 0.766
0.75 0.744
0.80 0.735
0.85 0.726
0.90 0.726
0.95 0.726
1.00 0.726
```
The model performs best at a 0.5 threshold, confirming it is the optimal choice for this context. We can visualize how accuracy changes with different thresholds:
```python
import matplotlib.pyplot as plt

plt.plot(thresholds, scores)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy at Different Thresholds')
plt.show()
```
Scikit-learn Accuracy
We can simplify this evaluation by using Scikit-Learn's accuracy_score function:
```python
from sklearn.metrics import accuracy_score

thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```
```python
0.00 0.274
0.05 0.509
0.10 0.591
0.15 0.666
0.20 0.710
0.25 0.739
0.30 0.760
0.35 0.772
0.40 0.785
0.45 0.793
0.50 0.803
0.55 0.801
0.60 0.795
0.65 0.786
0.70 0.766
0.75 0.744
0.80 0.735
0.85 0.726
0.90 0.726
0.95 0.726
1.00 0.726
```
Dummy Model Accuracy
The dummy model (which predicts all customers as non-churners) achieves an accuracy of 73%, even though it doesn’t distinguish between churning and non-churning customers. This reveals the limitations of accuracy as a metric, especially with imbalanced datasets.
```python
from collections import Counter

# Distribution of predictions
Counter(y_pred >= 1.0)  # Output: Counter({False: 1409})

# Distribution of actual values
Counter(y_val)  # Output: Counter({0: 1023, 1: 386})

1023 / 1409  # Output: 0.7260468417317246
y_val.mean()  # Output: 0.2739531582682754
1 - y_val.mean()  # Output: 0.7260468417317246
```
With only 27% churners, accuracy can be deceptive, as predicting everyone as non-churners already gives a high score.

Alternative Metrics for Imbalanced Datasets
In cases like this, it’s important to consider other metrics:

Precision: Measures the proportion of true positives among all positive predictions.
Recall: Measures the proportion of true positives among all actual positives.
F1-Score: The harmonic mean of precision and recall.
AUC-ROC: Measures the ability to distinguish between classes at various thresholds.
Choosing the best metric depends on the problem's goals and whether minimizing false positives or false negatives is more important.
