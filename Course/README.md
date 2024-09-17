# Machine Learning Zoomcamp

This course is designed for understand the fundamentals of machine learning and learn how to use key ML frameworks and tools. The only prerequisite for the course is prior programming experience (at least one year) and familiarity with the command line.

The course is structured into two parts:


![image](https://github.com/user-attachments/assets/22bf5e3b-d24f-4b54-82ae-a5d567292a5b)

Part 1 focuses on machine learning algorithms implemented in Python, including Linear Regression, Classification, Decision Trees, Ensemble Learning, and Neural Networks. 

![image](https://github.com/user-attachments/assets/4ea6d9cf-d038-4052-858f-9761fd93f495)

Part 2 is centered around deploying models using popular frameworks such as Flask, TensorFlow, and Kubernetes.

<!--
### Syllabus 

- [Introduction to Machine Learning](#1-introduction-to-machine-learning)
- [Machine Learning for Regression](#2-machine-learning-for-regression)
- [Machine Learning for Classification](#3-machine-learning-for-classification)
- [Evaluation Metrics for Classification](#4-evaluation-metrics-for-classification)
- [Deploying Machine Learning Models](#5-deploying-machine-learning-models)
- [Decision Trees and Ensemble Learning](#6-decision-trees-and-ensemble-learning)
- [Neural Networks and Deep Learning](#8-neural-networks-and-deep-learning)
- [Serverless Deep Learning](#9-serverless-deep-learning)
- [Kubernetes and TensorFlow Serving](#10-kubernetes-and-tensorflow-serving)

## Taking the course

### 2024 cohort

We start the course again in September 2024

* 16 September, 17:00 Berlin time
* [Sign up here](https://airtable.com/shryxwLd0COOEaqXo)
* Register at [DataTalks.Club](https://DataTalks.Club/slack.html) and join the `#course-ml-zoomcamp` channel 
* Join the [course telegram channel](https://t.me/mlzoomcamp)
* Subscribe to the [public google calendar](https://calendar.google.com/calendar/?cid=cGtjZ2tkbGc1OG9yb2lxa2Vwc2g4YXMzMmNAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ) (subscribing works from desktop only)
* [Tweet about it](https://ctt.ac/XZ6b9)
* If you have questions, check [FAQ](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
* All the materials specific to the 2024 will be in [the 2024 cohort folder](./cohorts/2024/) 


### Self-paced mode

You can take the course at your own pace. All the materials are freely available, and you can start learning at any time.

To take the best out of this course, we recommened this:

* Register at [DataTalks.Club](https://DataTalks.Club) and join the `#course-ml-zoomcamp` channel 
* For each module, watch the videos and work through the code
* If you have any questions, ask them in the `#course-ml-zoomcamp` channel in Slack
* Do homework. There are solutions, but we advise to first attempt the homework yourself, and after that check the solutions
* Do at least one project. Two is better. Only this way you can make sure you're really learning. If you need feedback, use the `#course-ml-zoomcamp` channel

Of course, you can take each module independently.

### Prerequisites

* Prior programming experience (at least 1+ year)
* Being comfortable with command line 
* No prior exposure to machine learning is required

Nice to have but not mandatory

* Python (but you can learn it during the course)
* Prior exposure to linear algebra will be helpful (e.g. you studied it in college but forgot)

## Asking questions

The best way to get support is to use [DataTalks.Club's Slack](https://datatalks.club/slack.html). Join the [`#course-ml-zoomcamp`](https://app.slack.com/client/T01ATQK62F8/C0288NJ5XSA) channel.

To make discussions in Slack more organized:

* Follow [these recommendations](asking-questions.md) when asking for help
* Read the [DataTalks.Club community guidelines](https://datatalks.club/slack/guidelines.html)

> We encourage [Learning in Public](learning-in-public.md)
-->

# Course Outline

## 1. Introduction to Machine Learning
- 1.1 Introduction to Machine Learning
- 1.2 ML vs Rule-Based Systems
- 1.3 Supervised Machine Learning
- 1.4 CRISP-DM
- 1.5 Model Selection Process
- 1.6 Setting up the Environment
- 1.7 Introduction to NumPy
- 1.8 Linear Algebra Refresher
- 1.9 Introduction to Pandas
- 1.10 Summary
- 1.11 Homework

## 2. Machine Learning for Regression
- 2.1 Car price prediction project
- 2.2 Data preparation
- 2.3 Exploratory data analysis
- 2.4 Setting up the validation framework
- 2.5 Linear regression
- 2.6 Linear regression: vector form
- 2.7 Training linear regression: Normal equation
- 2.8 Baseline model for car price prediction project
- 2.9 Root mean squared error
- 2.10 Using RMSE on validation data
- 2.11 Feature engineering
- 2.12 Categorical variables
- 2.13 Regularization
- 2.14 Tuning the model
- 2.15 Using the model
- 2.16 Car price prediction project summary
- 2.17 Explore more
- 2.18 Homework

## 3. Machine Learning for Classification
- 3.1 Churn prediction project
- 3.2 Data preparation
- 3.3 Setting up the validation framework
- 3.4 EDA
- 3.5 Feature importance: Churn rate and risk ratio
- 3.6 Feature importance: Mutual information
- 3.7 Feature importance: Correlation
- 3.8 One-hot encoding
- 3.9 Logistic regression
- 3.10 Training logistic regression with Scikit-Learn
- 3.11 Model interpretation
- 3.12 Using the model
- 3.13 Summary
- 3.14 Explore more
- 3.15 Homework

## 4. Evaluation Metrics for Classification
- 4.1 Evaluation metrics: session overview
- 4.2 Accuracy and dummy model
- 4.3 Confusion table
- 4.4 Precision and Recall
- 4.5 ROC Curves
- 4.6 ROC AUC
- 4.7 Cross-Validation
- 4.8 Summary
- 4.9 Explore more
- 4.10 Homework

## 5. Deploying Machine Learning Models
- 5.1 Intro / Session overview
- 5.2 Saving and loading the model
- 5.3 Web services: introduction to Flask
- 5.4 Serving the churn model with Flask
- 5.5 Python virtual environment: Pipenv
- 5.6 Environment management: Docker
- 5.7 Deployment to the cloud: AWS Elastic Beanstalk
- 5.8 Summary
- 5.9 Explore more
- 5.10 Homework

## 6. Decision Trees and Ensemble Learning
- 6.1 Credit risk scoring project
- 6.2 Data cleaning and preparation
- 6.3 Decision trees
- 6.4 Decision tree learning algorithm
- 6.5 Decision trees parameter tuning
- 6.6 Ensemble learning and random forest
- 6.7 Gradient boosting and XGBoost
- 6.8 XGBoost parameter tuning
- 6.9 Selecting the best model
- 6.10 Summary
- 6.11 Explore more
- 6.12 Homework

## 7. Midterm Project
Putting everything we've learned so far in practice!

## 8. Neural Networks and Deep Learning
- 8.1 Fashion classification
- 8.1b Setting up the Environment on Saturn Cloud
- 8.2 TensorFlow and Keras
- 8.3 Pre-trained convolutional neural networks
- 8.4 Convolutional neural networks
- 8.5 Transfer learning
- 8.6 Adjusting the learning rate
- 8.7 Checkpointing
- 8.8 Adding more layers
- 8.9 Regularization and dropout
- 8.10 Data augmentation
- 8.11 Training a larger model
- 8.12 Using the model
- 8.13 Summary
- 8.14 Explore more
- 8.15 Homework

## 9. Serverless Deep Learning
- 9.1 Introduction to Serverless
- 9.2 AWS Lambda
- 9.3 TensorFlow Lite
- 9.4 Preparing the code for Lambda
- 9.5 Preparing a Docker image
- 9.6 Creating the lambda function
- 9.7 API Gateway: exposing the lambda function
- 9.8 Summary
- 9.9 Explore more
- 9.10 Homework

## 10. Kubernetes and TensorFlow Serving
- 10.1 Overview
- 10.2 TensorFlow Serving
- 10.3 Creating a pre-processing service
- 10.4 Running everything locally with Docker-compose
- 10.5 Introduction to Kubernetes
- 10.6 Deploying a simple service to Kubernetes
- 10.7 Deploying TensorFlow models to Kubernetes
- 10.8 Deploying to EKS
- 10.9 Summary
- 10.10 Explore more
- 10.11 Homework

## 11. KServe (optional)
- 11.1 Overview
- 11.2 Running KServe locally
- 11.3 Deploying a Scikit-Learn model with KServe
- 11.4 Deploying custom Scikit-Learn images with KServe
- 11.5 Serving TensorFlow models with KServe
- 11.6 KServe transformers
- 11.7 Deploying with KServe and EKS
- 11.8 Summary
- 11.9 Explore more

## Capstone Project 1



## Capstone Project 2


## Supporters and partners

<p align="center">
  <a href="https://saturncloud.io/">
    <img height="120" src="https://github.com/DataTalksClub/llm-zoomcamp/raw/main/images/saturn-cloud.png">
  </a>
</p>



