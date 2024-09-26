# ML Zoomcamp 2024: Introduction to Machine Learning  

### Part 1

## What is Machine Learning

The concept of Machine Learning (ML) is illustrated through an example of predicting car prices. Data, including features such as year and mileage, is used by the ML model to learn and identify patterns. The target variable, in this case, is the car's price.

New data, which lacks the target variable, is then provided to the model to predict the price.

In summary, ML involves extracting patterns from data, which is categorized into two types:
- **Features**: Information about the object.
- **Target**: The property to be predicted for unseen objects.

New feature values are inputted into the model, which generates predictions based on the patterns it has learned. This is an overview of what has been learned from the ML course by Alexey Grigorev ([ML Zoomcamp](http://mlzoomcamp.com)). All images in this post are sourced from the course material. Images in other posts may also be derived from this material.

### What is Machine Learning?

Machine Learning (ML) is explained as the process of training a model using features and target information to predict unknown object targets. In other words, ML is about extracting patterns from data, which includes features and targets.

To understand ML, it is important to differentiate between the following terms:
- **Features**: What is known about an object. In this example, it refers to the characteristics of a car. A feature represents an object's attribute in various forms, such as numbers, strings, or more complex formats (e.g., location information).
- **Target**: The aspect to be predicted. The term "label" is also used in some sources. During training, a labeled dataset is used since the target is known. For example, datasets of cars with known prices are used to predict prices for other cars with unknown values.
- **Model**: The result of the training process, which encompasses the patterns learned from the training data. This model is utilized later to make predictions about the target variable based on the features of an unknown object.

### Training and Using a Model

- **Train a Model**: The training process involves the extraction of patterns from the provided training data. In simpler terms, features are combined with the target, resulting in the creation of the model.

- **Use a Model**: Training alone does not make the model useful. The benefit is realized through its application. By applying the trained model to new data (without targets), predictions for the missing information (e.g., price) are obtained. Therefore, features are used during prediction, while the trained model is applied to generate predictions for the target variable.

### Part 2

## Machine Learning vs Rule Bases

## Differences Between ML and Rule-Based Systems: Example of a Spam Filter

### Rule-Based Systems

Traditional rule-based systems rely on a set of characteristics (such as keywords and email length) to determine whether an email is spam. As spam emails evolve, the system must be updated, which becomes increasingly complex due to the difficulty of maintaining and modifying the code as the system expands.

### Machine Learning Approach

The problem of spam filtering can be addressed using machine learning through the following steps:

1. **Data Collection**  
   Emails from the spam folder and inbox are gathered to provide examples of both spam and non-spam emails.

2. **Feature Definition and Calculation**  
   Features for the ML model can be initially defined using the rules from rule-based systems. The target variable for each email is determined based on its source (spam folder or inbox).  
   Each email is then encoded into feature values and a target label.

3. **Model Training and Application**  
   A machine learning algorithm is applied to the encoded emails to build a model capable of predicting whether a new email is spam or not. The predictions generated are probabilities, and a threshold must be defined to classify emails as spam or not spam.

### Rule-Based Systems

In a rule-based system, rules are defined to differentiate between ham and spam emails. Initially, the rules work effectively, but over time, adjustments become necessary. This continuous need for reconfiguration creates a maintenance challenge, making the system increasingly difficult to manage.

### Machine Learning

To implement a spam filter using machine learning, the following steps are taken:

1. **Data Collection**  
   Data is collected using the “SPAM” button in the email system.

2. **Feature Definition and Extraction**  
   Features are created, starting with rules used in rule-based systems. Examples of features include:
   - Length of title > 10? true/false
   - Length of body > 10? true/false
   - Sender “promotions@online.com”? true/false
   - Sender “hpYOSKmL@test.com”? true/false
   - Sender domain “test.com”? true/false
   - Description contains “deposit”? true/false

   These features are binary, and each email can be encoded as a binary vector, such as [1, 1, 0, 0, 1, 1]. Each email also has a label/target (spam = 1, no-spam = 0), which represents the desired output.

3. **Training**  
   The data is used to train the model, a process often referred to as fitting the model. During training, a complex system of equations with numerous parameters is solved. Features are adjusted relative to each other to achieve the correct classification, with the trained model determining the weights needed to accurately classify emails as spam (1) or not spam (0). The model provides probabilities for the correct label.

4. **Applying the Model**  
   When applied to new datasets, the model generates a probability indicating whether the email is spam. A threshold (e.g., 0.5) is used to make the final classification decision, with probabilities greater than or equal to 0.5 categorized as spam.

### Part 3

## Supervised Machine Learning Overview

In Supervised Machine Learning (SML), labels are always associated with specific features. The model is trained on these labeled examples, and subsequently, predictions can be made on new, unseen features. This training process involves teaching the model by using a feature matrix \( X \) and target values \( y \), with \( X \) representing observations or objects (rows) and features (columns), and \( y \) being the vector containing the target information.

The model can be represented as a function \( g \) that takes the feature matrix \( X \) as input and attempts to predict values as close as possible to the target values \( y \). The process of determining the function \( g \) is known as training.

### Types of SML Problems

- **Regression:** The output is a continuous number, such as the price of a car.
- **Classification:** The output is a category, such as identifying an email as spam or not spam.
  - **Binary Classification:** There are two categories.
  - **Multiclass Classification:** There are more than two categories.
- **Ranking:** The output consists of scores associated with items, commonly used in recommender systems.

In summary, SML involves teaching a model by presenting various examples to it, with the goal of deriving a function that can take the feature matrix as input and make predictions as closely aligned with the target values \( y \) as possible.

### Approaches to Software Solutions

Several approaches exist for addressing problems with software. These include:
- **Classical Approach:** Solutions are hard-coded.
- **AI Approaches:**
  - **Knowledge-Based Systems:** These are divided into rule-based systems and case-based reasoning.
  - **Machine Learning (ML):** This approach allows systems to learn from experience without explicit programming. ML focuses on developing applications that can access and learn from data independently.

Different problems are addressed by ML, including:
- **Regression:** Predicting continuous values, such as prices.
- **Classification:** Predicting labels to distinguish between different classes.
- **Clustering:** Identifying groups or patterns in data without predefined group labels.

### Learning Strategies

Depending on the problem type, various learning strategies can be employed:
- **Supervised Learning**
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**
- **Active Learning**

This overview provides insight into where Supervised Learning fits within the broader context of machine learning approaches. For further details on other approaches, consult relevant literature.

### Definition of Supervised Machine Learning

In supervised machine learning, the model is guided by the training data, which consists of examples paired with target values (e.g., car prices). The model learns from these examples, extracting patterns and generalizing them to new examples.

- **Rows:** Observations or objects for prediction.
- **Columns:** Features of each observation/object.
- **Feature Matrix (X):** A two-dimensional array containing all features.
- **Target Variable (y):** A one-dimensional array containing target values.

The formal definition of supervised machine learning is expressed as \( g(X) \approx y \), where:
- \( X \): Feature matrix
- \( y \): Target variable
- \( g \): Model that takes \( X \) and approximates \( y \)

Training aims to develop the function \( g \). Although the model (function \( g \)) may not always predict the exact target variable, the goal is to approximate \( y \) as closely as possible.

### Types of Supervised Machine Learning

- **Regression:**
  - Example: Predicting the price of a car or house.
  - \( g \) predicts a continuous number within the range of \(-\infty\) to \(+\infty\).

- **Classification:**
  - Example: Identifying a picture as a car or classifying mail as spam.
  - \( g \) predicts a category or label.
  - **Subtypes of Classification:**
    - **Multiclass Classification:** Distinguishing among several classes (e.g., cat, dog, car).
    - **Binary Classification:** Distinguishing between two classes (e.g., spam vs. not spam).

- **Ranking:**
  - Used to rank items, such as in recommender systems, where items are scored and the top values are highlighted based on potential interest. Google's search engine operates similarly by ranking search results.

### Part 4

# CRISP-DM: Cross-Industry Standard Process for Data Mining

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is an open standard process model that describes common approaches used by data mining experts. Conceived in 1996 and developed as a European Union project under the ESPRIT funding initiative in 1997, it was led by five companies: Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation, and OHRA (an insurance company).

## CRISP-DM Process Overview

![image](https://github.com/user-attachments/assets/8e8af0c3-49cf-45a2-9754-016e536a96bb)


The CRISP-DM process is an iterative methodology consisting of six steps:

1. **Business Understanding**
   - The business problem is identified.
   - Available data sources are detected.
   - Requirements, premises, and conditions are specified.
   - Risks and uncertainties are clarified.
   - The importance of the problem is assessed.
   - Potential solutions are explored.
   - Success metrics for the project are defined, including a Cost-Benefit Analysis.
   - The necessity of Machine Learning (ML) for the project is evaluated.

2. **Data Understanding**
   - Available data sources are analyzed.
   - Data is collected and analyzed.
   - Missing data is identified and assessed.
   - Data reliability and sufficiency are evaluated.
   - Decisions are made regarding the need for additional data.

3. **Data Preparation (Feature Engineering)**
   - Data is transformed into a suitable format for ML algorithms.
   - Features are extracted from raw data.
   - Data is cleaned and noise is removed.
   - Pipelines for transforming raw data into clean data are built.
   - Data is converted into a tabular format required for ML models.

   Feature Engineering is a critical component of ML projects. As Andrew Ng, Professor at Stanford University, stated, “Coming up with features is difficult, time-consuming, and requires expert knowledge. ‘Applied Machine Learning’ is essentially feature engineering.”

4. **Modeling**
   - Models are trained using different algorithms.
   - Various models such as Logistic Regression, Decision Trees, and Neural Networks are tested.
   - Model parameters are selected and adjusted.
   - Model quality is improved and the best model is chosen.
   - Adjustments to data preparation may be required, including adding new features and fixing data issues.

   A key aspect to remember is that model quality is significantly dependent on data quality. The principle of “Garbage in, Garbage out” must be considered.

5. **Evaluation**
   - The model's performance in solving the business problem is measured.
   - Metrics are assessed to determine if goals have been achieved (e.g., reducing spam by 50%).
   - Retrospective evaluation is conducted to determine if the goals were achievable and if the right metrics were used.
   - Decisions are made on whether to adjust the goals, expand the model’s deployment, or halt the project.

6. **Deployment**
   - The model is deployed to production after online evaluation (live users) and proper monitoring.
   - The model is rolled out to all users.
   - Quality and maintainability are ensured.
   - Scalability and other engineering practices are addressed.
   - A final report is created to summarize the project.

## Iteration

- The ML process often requires multiple iterations:
  - Begin with simple models.
  - Incorporate feedback to refine the model.
  - Make continuous improvements.

Overall, CRISP-DM helps in organizing and managing ML projects efficiently, with a focus on iterative improvement and integration of feedback.

### Part 5

# Model Selection Process

## Overview

The model selection process involves choosing the most suitable model from various options, such as Logistic Regression, Decision Trees, Neural Networks, and others. To ensure reliable model performance, the dataset is typically divided into training, validation, and test sets. 

## Steps to Get Model Performance

1. **Split the Dataset**
   - The dataset is split into training (60%), validation (20%), and test (20%) sets.

2. **Train the Models**
   - Models are trained using the training dataset.

3. **Evaluate the Models**
   - The trained models are evaluated on the validation dataset to determine their performance.

4. **Select the Best Model**
   - The model that performs best on the validation dataset is selected.

5. **Apply the Best Model to the Test Dataset**
   - The selected model is then applied to the test dataset to assess its performance on unseen data.

6. **Compare Performance Metrics**
   - The performance metrics of the validation and test datasets are compared to ensure that the model generalizes well.

## Multiple Comparison Problem

The multiple comparisons problem (MCP) occurs when different models are compared on the same validation dataset. A model may appear to perform well simply by chance, as all models are probabilistic. To mitigate MCP, the test set is used to verify that the selected model is indeed the best.

## Training – Validation – Test – Split

To address MCP, the dataset should be divided into three distinct sets:

- **Training Set:** Used to train the models.
- **Validation Set:** Used to evaluate model performance during training.
- **Test Set:** Used to confirm the model's performance on completely unseen data.

The following process is typically followed:

1. **Train Models**
   - Various models are trained using the training dataset.

2. **Evaluate Models**
   - Models are evaluated using the validation dataset to identify the best performer.

3. **Apply Model to Test Dataset**
   - The best-performing model is tested on the test dataset.

4. **Compare Metrics**
   - Performance metrics from the validation and test datasets are compared.

The model selection process involves:

1. Splitting the dataset into training, validation, and test sets.
2. Training models on the training dataset.
3. Evaluating models using the validation dataset.
4. Selecting the best model based on validation performance.
5. Testing the selected model on the test dataset.
6. Comparing the performance metrics from both validation and test datasets.

## Alternative Approach

To avoid wasting the validation dataset, it can be reused. The steps for this approach are:

1. **Split the Dataset**
   - The dataset is divided into training (60%), validation (20%), and test (20%) sets.

2. **Train Initial Models**
   - Initial models are trained on the training dataset.

3. **Evaluate on Validation Dataset**
   - The models are applied to the validation dataset, and their performance is evaluated.

4. **Select the Best Model**
   - The best-performing model is chosen based on validation results.

5. **Combine Training and Validation Datasets**
   - The training and validation datasets are combined to create a new, larger dataset.

6. **Retrain the Model**
   - The selected model is retrained using the combined dataset.

7. **Apply to Test Dataset**
   - The retrained model is applied to the test dataset to evaluate its performance on new data.

Combining the training and validation datasets can potentially enhance the model's performance by allowing it to learn from a larger dataset. The final evaluation on the test dataset provides a reliable measure of the model’s ability to generalize.

It should be noted that the effectiveness of the alternative approach may vary depending on the dataset's characteristics and the initial models' performance. Careful evaluation and experimentation are essential to determine the most appropriate approach for the machine learning task.

In summary, the model selection process is vital for machine learning, involving careful assessment and selection of the best model based on unseen data. The alternative approach of combining datasets can be an effective strategy to improve performance and generalization.

### Part 6

# Setting Up the Environment

## Requirements

To prepare your environment, you need:

- **Python 3.10** (Note: Videos use Python 3.8)
- **NumPy, Pandas, and Scikit-Learn** (latest versions)
- **Matplotlib and Seaborn**
- **Jupyter notebooks**

## Ubuntu 22.04 on AWS

- Refer to [this video](https://www.youtube.com/watch?v=IXSiYkP23zo) for a complete environment configuration on an AWS EC2 instance.
- Adjust the instructions to clone the relevant repository instead of the MLOps one.
- You can also use these instructions for setting up local Ubuntu.

### Note for WSL

- Most instructions from the video apply to WSL as well.
- For Docker, install Docker Desktop on Windows; it will be automatically used in WSL. No need to install `docker.io`.

## Anaconda and Conda

Using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended:

- **Anaconda**: Includes everything needed.
- **Miniconda**: A lighter version containing only Python.

Follow the installation instructions on the respective websites.

### Part 7

# NumPy: A Comprehensive Overview

NumPy is a powerful library in Python used for numerical computing, providing support for multi-dimensional arrays and a variety of mathematical functions. It is an essential tool for data analysis, scientific computing, and machine learning.

## 1. Creating Arrays

NumPy allows the creation of arrays in various ways:

- **From Python Lists**: You can create a NumPy array directly from a Python list using `np.array()`.
    ```python
    import numpy as np
    arr = np.array([1, 2, 3])
    ```

- **Using Built-in Functions**: Functions such as `np.zeros()`, `np.ones()`, and `np.arange()` enable the creation of arrays initialized with zeros, ones, or a range of values.
    ```python
    zeros_array = np.zeros((2, 3))  # 2x3 array of zeros
    ones_array = np.ones((3, 2))     # 3x2 array of ones
    range_array = np.arange(10)       # Array with values from 0 to 9
    ```

- **Using Random Generation**: The `numpy.random` module provides methods for generating arrays filled with random values, useful for testing and simulations.
    ```python
    random_integers = np.random.randint(10, size=5)        # 1D array of random integers
    random_floats = np.random.random((3, 4))               # 2D array of random floats
    random_normal = np.random.normal(size=(2, 3, 2))       # 3D array from standard normal distribution
    ```

## 2. Element-wise Operations

NumPy supports element-wise operations, allowing mathematical operations to be performed directly on arrays without explicit loops. This includes:

- **Addition and Subtraction**:
    ```python
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    result_add = arr1 + arr2  # Element-wise addition
    result_sub = arr1 - arr2  # Element-wise subtraction
    ```

- **Multiplication and Division**:
    ```python
    result_mul = arr1 * arr2  # Element-wise multiplication
    result_div = arr1 / arr2  # Element-wise division
    ```

- **Mathematical Functions**: NumPy provides numerous built-in functions like `np.sin()`, `np.exp()`, etc., that operate element-wise.
    ```python
    arr = np.array([0, np.pi/2, np.pi])
    result_sin = np.sin(arr)  # Sine of each element
    result_exp = np.exp(arr)  # Exponential of each element
    ```

## 3. Comparison Operations

NumPy enables comparison operations between array elements, resulting in boolean arrays that can be used for filtering or conditional assignments:

- **Basic Comparisons**:
    ```python
    arr = np.array([1, 2, 3, 4, 5])
    result_comp = arr > 3  # Output: [False False False True True]
    ```

- **Element-wise Comparisons**: You can compare elements of multiple arrays:
    ```python
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([3, 2, 1])
    result_eq = arr1 == arr2  # Output: [False True False]
    ```

- **Logical Combinations**: Combine comparisons with logical operators:
    ```python
    result_combined = (arr > 2) & (arr < 5)  # Output: [False False True True False]
    ```

NumPy is a fundamental library for anyone involved in scientific computing or data analysis in Python. Its ability to handle arrays and perform complex operations efficiently makes it a go-to tool for developers and researchers alike.

### Part 8

# Linear Algebra Refresher

## Part 1: Introduction to Linear Algebra
Linear algebra is a branch of mathematics that focuses on vector spaces and linear mappings between them. It encompasses various concepts essential for understanding more complex mathematical theories and applications, especially in fields like machine learning and data analysis.

### Key Concepts:
1. **Vectors**: An ordered array of numbers representing a point in space.
2. **Matrices**: A rectangular array of numbers organized in rows and columns, representing linear transformations.

## Part 2: Fundamental Operations
### Vector Operations:
- **Addition**: Vectors of the same dimension can be added component-wise.
- **Scalar Multiplication**: Each component of the vector is multiplied by a scalar.
- **Dot Product**: A scalar resulting from the sum of the products of corresponding components of two vectors.

### Matrix Operations:
- **Addition**: Similar to vector addition, matrices of the same dimensions can be added.
- **Scalar Multiplication**: Each element of the matrix is multiplied by a scalar.
- **Matrix Multiplication**: The dot product of rows and columns, requiring the number of columns in the first matrix to equal the number of rows in the second.

## Part 3: Special Matrix Types

### Identity Matrix
- The **identity matrix** (denoted as \( I \)) is a square matrix with ones on the main diagonal and zeros elsewhere.
- **Mathematical Representation**:
  ![image](https://github.com/user-attachments/assets/dfae5539-5cbb-4246-971f-65e4a2d76b62)

- **Properties**:
  - Acts as a neutral element in matrix multiplication: ![image](https://github.com/user-attachments/assets/0cdc23aa-6ad0-4c94-b63a-f86085eeb08d)

- **Applications**:
  - Used in neural networks as initial weight matrices to prevent overfitting and aid convergence.

**Python Implementation**:
python

```
import numpy as np
```

# Creating a 3x3 identity matrix
I = np.eye(3)  
# Inverse Matrix, Eigenvalues, and Determinants in Linear Algebra

## Inverse Matrix
The inverse of a matrix \( U \) (denoted \( U^{-1} \)) satisfies the equation:
![image](https://github.com/user-attachments/assets/d54cc42f-bb82-4a87-8c8a-b78c899b355d)


### Conditions:
- \( U \) must be square and invertible (i.e., the determinant \( |U| \neq 0 \)).

### Formula:
![image](https://github.com/user-attachments/assets/51ca46ff-9c69-4eaa-834c-04d6ebef3354)


### Applications:
- Solving linear equations.
- Optimizing algorithms in machine learning.

### Python Implementation:
python

```
import numpy as np
```
# Defining a square matrix
V = np.array([
    [1, 1, 2],
    [0, 0.5, 1],
    [0, 3, 1]
])

# Calculating the inverse of matrix V
V_inv = np.linalg.inv(V)  
# Eigenvalues, Eigenvectors, and Determinants in Linear Algebra

## Eigenvalues and Eigenvectors
Eigenvalues (\( \lambda \)) and eigenvectors (\( v \)) are crucial for understanding matrix behavior, expressed as:
![image](https://github.com/user-attachments/assets/aa38ec1c-9f7b-442e-ac06-48a2182641d2)


### Applications:
- Used in dimensionality reduction techniques such as Principal Component Analysis (PCA) and spectral clustering.

## Determinants
The determinant of a matrix \( A \) (denoted \( |A| \)) provides insights into the matrix's scaling factor and invertibility.

### Importance:
- Determines if a matrix is singular or invertible.
- Useful in solving linear equations and finding eigenvalues.

### Calculation:
- Various methods exist based on matrix size (e.g., LU decomposition for larger matrices).

Understanding these concepts in linear algebra provides a strong foundation for more advanced mathematical applications in machine learning and data analysis. Mastery of identity matrices, inverse matrices, eigenvalues, eigenvectors, and determinants enhances one's ability to tackle complex problems effectively. These mathematical tools serve as essential components in the development of algorithms and data manipulation strategies used in various applications.

### Part 9

# Introduction to Pandas

Pandas is a powerful Python library widely used for data analysis and manipulation. It provides flexible data structures and functions to work with structured data, making it an essential tool for data scientists and analysts.

## 1. DataFrame Basics

A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns). It can be created from various data sources, such as dictionaries, lists, or external files like CSV.

### Example DataFrame

Consider the following DataFrame representing car information:

| Make    | Model      | Year | Engine HP | Engine Cylinders | Transmission Type | Vehicle_Style | MSRP  |
|---------|------------|------|-----------|-------------------|-------------------|----------------|-------|
| Nissan  | Stanza     | 1991 | 138       | 4                 | MANUAL            | sedan          | 2000  |
| Hyundai | Sonata     | 2017 | NaN       | 4                 | AUTOMATIC         | Sedan          | 27150 |
| Lotus   | Elise      | 2010 | 218       | 4                 | MANUAL            | convertible     | 54990 |
| GMC     | Acadia     | 2017 | 194       | 4                 | AUTOMATIC         | 4dr SUV        | 34450 |
| Nissan  | Frontier   | 2017 | 261       | 6                 | MANUAL            | Pickup         | 32340 |

## 2. Data Filtering

Filtering refers to the process of selecting specific rows or columns from a DataFrame based on certain conditions. In Pandas, we can use various techniques to filter our data.

### Boolean Indexing

A common technique is to use boolean indexing, which involves creating a boolean mask specifying the conditions for data selection.

For example, to filter out all rows where the year is greater than or equal to 2015:

```python
condition = df.Year >= 2015
filtered_df = df[condition]  # or df[df.Year >= 2015]
```

### Using the .query() Method
Another useful technique is the .query() method, which allows us to filter rows using a string expression similar to SQL:
```
filtered_df = df.query('Year >= 2015')
```

### Filtering by Make
To filter for cars made by Nissan:
```
df[df.Make == 'Nissan']
```
### Combining Conditions
We can combine multiple conditions. For example, to get all Nissans made after the year 2015:
```
df[(df.Make == 'Nissan') & (df.Year > 2015)]
```
By using these filtering techniques, we can easily extract the data needed for further analysis or computations, especially with large datasets.

## 3. String Operations
Pandas provides string operations that are not available in NumPy, which primarily focuses on numerical data.

Example DataFrame and Vehicle_Style Column
The Vehicle_Style column may have inconsistent formatting. We can standardize it by converting all text to lowercase and replacing spaces with underscores.
```
df['Vehicle_Style'] = df['Vehicle_Style'].str.lower().str.replace(' ', '_')
```
Summary of Operations
We can also summarize numerical columns using various functions:
```
df.MSRP.min()  # Minimum MSRP
df.MSRP.max()  # Maximum MSRP
df.MSRP.mean()  # Average MSRP
```
## 4. Descriptive Statistics
The describe() function provides a summary of numerical columns, including count, mean, standard deviation, min, max, and quantiles.
```
df.describe()  # Summary of all numerical columns
df.MSRP.describe()  # Summary for a specific column
```
To round the values for better readability:
```
df.describe().round(2)
```
## 5. Handling Categorical Columns
Unique Values
To count the number of unique values in a column:
```
df.Make.nunique()  # Unique makes
df.nunique()  # Unique values for all columns
```
Unique Values List
To see the unique values in a specific column:
```
df.Year.unique()
```
## 6. Missing Values
Handling missing values is crucial. The isnull() function returns a boolean DataFrame indicating missing values:
```
df.isnull().sum()  # Summarizes the number of missing values per column
```
## 7. Grouping Data
Grouping allows us to summarize data. For example, to get the average MSRP for each transmission type:
```
df.groupby('Transmission Type').MSRP.mean()
```
This provides insights into how different groups compare regarding various metrics.
By using these techniques in Pandas, you can effectively analyze and manipulate data, leading to meaningful insights and informed decision-making.











