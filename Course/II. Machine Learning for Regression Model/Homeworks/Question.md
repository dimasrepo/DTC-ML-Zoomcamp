## Homework 2: Laptop Price Prediction

### Dataset

For this assignment, we will use the Laptops Price dataset from [Kaggle](https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset).

Our objective is to create a regression model to predict laptop prices using the column `'Final Price'`.

### Question 1

One of the columns contains missing values. Which column is it?

- `'ram'`
- `'storage'`
- `'screen'`
- `'final_price'`

### Question 2

What is the median (50th percentile) value for the variable `'ram'`?

- 8
- 16
- 24
- 32

### Data Preparation and Splitting

- Shuffle the dataset (use the filtered dataset you’ve created earlier), applying seed `42`.
- Split the dataset into training, validation, and test sets, with a 60%/20%/20% distribution.

Use the code provided in the lessons for reference.

### Question 3

- We need to handle missing values from the column identified in Q1.
- There are two options: fill missing values with 0 or with the mean of the column.
- For both approaches, train a linear regression model without regularization using the code from the lessons.
- When calculating the mean, make sure to use only the training set.
- Use the validation dataset to assess the model performance and compare the RMSE of each option.
- Round the RMSE values to two decimal places using `round(score, 2)`.

Which method provides the better RMSE?

- Fill with 0
- Fill with mean
- Both options are equally effective

### Question 4

Now, we’ll apply regularization to our linear regression model.

- Fill missing values with 0 for this question.
- Test different values of the regularization parameter `r` from the following list: `[0, 0.01, 0.1, 1, 5, 10, 100]`.
- Use the RMSE metric to evaluate the model on the validation set.
- Round the RMSE scores to two decimal places.

Which `r` value gives the best RMSE? If multiple values result in similar RMSE, select the smallest `r`.

- 0
- 0.01
- 1
- 10
- 100

### Question 5

We previously used seed 42 for data splitting. Let's explore how different seeds affect the model's performance.

- Test the following seed values: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- For each seed, perform the train/validation/test split with a 60%/20%/20% distribution.
- Fill missing values with 0 and train a linear regression model without regularization.
- For each seed, evaluate the model on the validation dataset and collect the RMSE values.
- Compute the standard deviation of the RMSE values using `np.std`.
- Round the result to three decimal places (`round(std, 3)`).

What is the standard deviation of the RMSE values?

- 19.176
- 29.176
- 39.176
- 49.176

> **Note**: Standard deviation measures how much the values differ from each other. A low standard deviation indicates that the values are similar, while a high standard deviation suggests more variation. If the standard deviation of RMSE values is low, it means that our model is stable.

### Question 6

- Split the dataset like before, but this time use seed `9`.
- Combine the training and validation sets.
- Fill missing values with 0 and train a model using `r = 0.001`.
- What is the RMSE on the test dataset?

Options:

- 598.60
- 608.60
- 618.60
- 628.60
