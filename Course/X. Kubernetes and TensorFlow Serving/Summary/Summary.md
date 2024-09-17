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
