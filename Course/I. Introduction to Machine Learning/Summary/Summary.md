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

## Summary

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
