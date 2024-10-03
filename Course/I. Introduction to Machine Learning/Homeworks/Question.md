## Homework: Laptop Price Analysis with Pandas

### Question 1: Pandas Version

What version of Pandas are you currently using?

You can retrieve this information by accessing the `__version__` attribute of Pandas.

### Data Acquisition

For this assignment, we'll work with the Laptop Price dataset. You can download it from the following link: [Laptops Price Dataset](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv).

Alternatively, you can open it in your browser and save it directly.

Once downloaded, use Pandas to load the dataset.

### Question 2: Number of Records

How many rows (records) are there in the dataset?

- 12
- 1000
- 2160
- 12160

### Question 3: Laptop Brands

How many unique laptop brands are represented in the dataset?

- 12
- 27
- 28
- 2160

### Question 4: Columns with Missing Values

How many columns in the dataset contain missing values?

- 0
- 1
- 2
- 3

### Question 5: Maximum Final Price of Dell Laptops

What is the highest final price for Dell laptops in this dataset?

- 869
- 3691
- 3849
- 3936

### Question 6: Median Screen Value Adjustment

1. First, calculate the median value of the `Screen` column.
2. Then, determine the most frequent value (mode) in the `Screen` column.
3. Use the `fillna` method to replace any missing values in `Screen` with the mode.
4. After filling missing values, recalculate the median for the `Screen` column.

Has the median changed after filling the missing values?

- Yes
- No

> **Hint**: You may use the `mode()` and `median()` functions to assist with these calculations.

### Question 7: Summation of Weights

1. Filter out all rows where the brand is "Innjoo".
2. Select only the columns `RAM`, `Storage`, and `Screen`.
3. Extract the underlying NumPy array from this filtered dataset. Let's refer to this array as `X`.
4. Perform matrix-matrix multiplication between the transpose of `X` and `X`. You can get the transpose using `X.T`. Let’s call this result `XTX`.
5. Calculate the inverse of `XTX`.
6. Create an array `y` with the following values: `[1100, 1300, 800, 900, 1000, 1100]`.
7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply this product by `y`. Let’s call the result `w`.
8. What is the sum of all elements in the `w` array?

- 0.43
- 45.29
- 45.58
- 91.30
