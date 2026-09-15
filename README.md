## Smart Manufacturing IoT Monitoring: Complete Technical Analysis


PART 1 — High-Level Overview

In a smart factory, internet-connected sensors collect measurements continuously across machines and lines. Raw sensor streams must be processed and passed into machine learning models to enable automated monitoring, failure prediction, and continuous energy optimization.

```text
+---------------------------------------------------------------------------------+
|                                RAW SENSOR STREAM                                |
|    [Temperature]      [Humidity]      [Vibration]      [Energy Consumption]     |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                                DATA PREPROCESSING                               |
|        Data Exploration ---> Missing Value Audit ---> Feature Selection         |
+---------------------------------------------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+------------------------------------------+ +------------------------------------------+
|            REGRESSION TASK               | |          CLASSIFICATION TASK             |
|  Target: Continuous Energy Consumption   | |   Target: High Energy Consumption (0/1)  |
|  Algorithm: Linear Regression            | |   Algorithm: Logistic Regression         |
|  Metrics: MAE, MSE, R^2                  | |   Metrics: Accuracy, Precision, Recall,  |
|                                          | |            F1-score                      |
+------------------------------------------+ +------------------------------------------+
```

The Analytical Pipeline
Dataset Ingestion: Raw sensor data (smart_manufacturing_data.csv) is loaded into memory as a structured pandas.DataFrame.
Data Exploration: Statistical summaries and schema inspections check data dimensions, variable types, and distribution traits.
Data Cleaning: Audits identify missing (NaN / null) values.
Feature Identification & Selection: Predictor features (X) are separated from target variables (y).
Train/Test Split: Data is split into training (model learning) and testing (out-of-sample evaluation) subsets.
Regression Pipeline: A continuous target (e.g., energy consumption or primary numeric metrics) is modeled via LinearRegression and evaluated using MAE, MSE, and R^2.
Classification Pipeline: Continuous metrics are converted into discrete operational flags (e.g., High Energy vs. Normal Energy using thresholding), then modeled via LogisticRegression and evaluated using Accuracy, Precision, Recall, and F1-Score.
Comparative Analysis: Regression and classification paradigms are compared to clarify why evaluation strategies differ by task type.

PART 2 — Dataset Architecture

The repository uses the smart manufacturing dataset file smart_manufacturing_data.csv.

Data Loading Syntax

```python
import pandas as pd

df = pd.read_csv("smart_manufacturing_data.csv")
```

pd: Standard alias for the pandas library.
pd.read_csv(): Parses a comma-separated text file, infers data types per column, and loads the data into an in-memory two-dimensional data structure.
df: The resulting pandas.DataFrame object.

Data Terminology

| Term | Definition | Example from smart_manufacturing_data.csv |
|---|---|---|
| Dataset | The complete collection of raw observations | The entire file smart_manufacturing_data.csv |
| DataFrame | Two-dimensional, tabular, memory-resident data structure | The variable df |
| Row / Sample | A single record representing one sensor reading moment | Row 0: ts = 1574518494.484, co = 0.00495, etc. |
| Column | A named vertical slice containing values of one metric | temp column containing all temperature readings |
| Feature (X) | Input attributes fed to an ML model | temp, humidity, co, lpg, smoke |
| Target (y) | Output attribute the model is trained to predict | light (binary regression/classification target) |

Telemetry Schema

Total Rows: 405,659 samples
Total Columns: 9 variables

| Column Name | Data Type | Physical/Logical Meaning | Role in Modeling |
|---|---|---|---|
| ts | float64 | Unix Epoch Timestamp (seconds since Jan 1 1970) | Identifier / Excluded feature |
| device | object (string) | MAC address / Unique ID of the sensor module | Identifier / Excluded feature |
| co | float64 | Carbon Monoxide Concentration (ppm) | Numerical Input Feature (X) |
| humidity | float64 | Relative Humidity (%) | Numerical Input Feature (X) |
| lpg | float64 | Liquefied Petroleum Gas Concentration (ppm) | Numerical Input Feature (X) |
| motion | bool / float64 | Binary Motion Sensor Detection state (0/1) | Input Feature (X) |
| smoke | float64 | Smoke Density Index (ppm) | Numerical Input Feature (X) |
| temp | float64 | Ambient Air Temperature (°C) | Numerical Input Feature (X) |
| light | bool / float64 | Light Sensor Status / Target Indicator | Target Variable (y) |

PART 3 — Data Exploration

The notebook performs exploratory data analysis (EDA) to understand distributions, structural shapes, and missingness before modeling.

1. Dimensionality Check

```python
print(df.shape)
```
Execution: Returns tuple (405659, 9).
Meaning: The dataset contains 405,659 rows and 9 columns.

2. Schema and Structural Inspection

```python
df.info()
```
Execution: Prints concise summaries including memory usage, non-null counts, column names, and pandas data types (float64, object, bool).
Meaning: Verifies whether data types were correctly parsed during file ingestion.

3. Quick Visual Verification

```python
df.head()
```
Execution: Displays the first 5 records of the DataFrame.
Meaning: Confirms data alignment, expected value ranges, and column header parsing.

4. Null Value Quantification

```python
df.isnull().sum()
```
Execution: df.isnull() generates a boolean mask where missing cells return True. .sum() totals True values column-by-column.
Meaning: Checks for data gaps resulting from sensor disconnects or dropped network packets.

5. Summary Statistics

```python
df.describe()
```
Execution: Computes descriptive statistics for numeric columns.

| Metric | Mathematical Meaning | Physical Interpretation in IoT Sensors |
|---|---|---|
| count | Total non-null records | Total operational readings collected |
| mean | Arithmetic average | Baseline operational value for the sensor |
| std | Standard deviation | Signal stability (high values indicate high noise or volatile spikes) |
| min | Minimum observed value | Lowest sensor measurement recorded |
| 25% | First quartile (Q1) | 25% of readings fall below this threshold |
| 50% | Median / Second quartile (Q2) | Midpoint value (robust against extreme outliers) |
| 75% | Third quartile (Q3) | 75% of readings fall below this threshold |
| max | Maximum observed value | Peak sensor measurement recorded |

PART 4 — Missing Value Audit

```python
# Missing Value Check
print(df.isnull().sum())
```

Technical Assessment
Observed Result: Every column returns 0 missing values across all 405,659 records.
Methodology: Because no null values are present, imputation functions (such as df.fillna()) or drop operations (df.dropna()) are not executed.
Importance of Checking: Calling .isnull().sum() is a mandatory prerequisite step. Scikit-learn estimators raise explicit ValueError runtime exceptions when presented with NaN (Not a Number) values during floating-point operations.
Leakage Prevention: When imputation is required (e.g., mean filling), parameters must be computed only on training subsets (X_train) and transformed onto test sets (X_test) to avoid Data Leakage (inadvertently introducing test dataset distribution statistics into training models).

PART 5 — Variable Classification and Encoding

Machine learning models require inputs in numeric matrix structures.

Numerical vs. Categorical Variables

```text
                              VARIABLES IN TELEMETRY DATA
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
          NUMERICAL VARIABLES                           CATEGORICAL VARIABLES
     Continuous scale measurements                   Discrete label descriptions
  (e.g., temp, humidity, co, lpg, smoke)                  (e.g., device MAC)
                    |                                             |
                    v                                             v
         Direct Input to Models                      Requires Encoding (e.g. One-Hot)
```

Numerical Variables: Measured on continuous quantitative scales. Arithmetic operations (addition, scalar multiplication) are mathematically valid.
Columns: temp, humidity, co, lpg, smoke, ts.
ML Compatibility: Can be fed directly into regression/classification algorithms.

Categorical Variables: Represent discrete groups, identifiers, or non-numeric strings.
Columns: device (e.g., "b8:27:eb:bf:9d:51").
ML Compatibility: Linear models cannot apply floating-point weights to raw strings. Categorical variables must be converted to numeric representations using methods like One-Hot Encoding (pd.get_dummies()) or Label Encoding before training.

PART 6 — Feature Selection and Extraction

Feature Isolation Syntax

```python
# Feature matrix selection
X = df[['co', 'humidity', 'lpg', 'smoke', 'temp']]

# Target vector selection (Regression Target: temp or light metric continuous proxy)
y = df['temp']  # Evaluated continuously for energy/thermal regression modeling
```

X (Feature Matrix): A 2D DataFrame containing N samples by M predictor features (405659 x 5).
y (Target Vector): A 1D Series of length N containing the outcome variable to be predicted (405659).

Operational Example of a Single Row

```python
# Sample Row Index 0
X_row0 = {'co': 0.00495, 'humidity': 51.0, 'lpg': 0.00765, 'smoke': 0.0204, 'temp': 22.7}
y_row0 = 22.7
```

Excluded Features and Data Leakage Prevention
ts (Timestamp): Excluded because raw epoch times cause models to learn time-dependent trends rather than physical relationships, which can lead to poor generalization on future data.
device (MAC address): Excluded to prevent the model from memorizing individual sensor behaviors, allowing it to generalize across arbitrary deployed sensor modules.
y (Target Column): Excluded from X to eliminate Data Leakage. Including the target inside the feature matrix gives the model direct access to the correct answer during training, causing artificial performance metrics (R^2 = 1.0) that fail in deployment.

PART 7 — Train/Test Partitioning

To measure a model's ability to generalize to new, unseen sensor data, the dataset is partitioned into non-overlapping training and testing subsets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

+-----------------------------------------------------------------------------------+
|                            FULL DATASET (405,659 Rows)                            |
+---------------------------------------------------------+-------------------------+
|                TRAINING SUBSET (80%)                    |    TEST SUBSET (20%)    |
|               X_train, y_train                          |     X_test, y_test      |
|               (324,527 Rows)                            |     (81,132 Rows)       |
+---------------------------------------------------------+-------------------------+

Parameter Breakdown
test_size=0.2: 20% of the data (81,132 rows) is reserved for testing, while 80% (324,527 rows) is used for training.
random_state=42: Sets the pseudo-random seed to ensure reproducible splits across runs.

Subset Shapes
X_train: (324527, 5)
X_test: (81132, 5)
y_train: (324527,)
y_test: (81132,)

Why Training and Testing Data Must Be Separated
Evaluating a model on data it was trained on can hide overfitting—where a model memorizes specific noise in the training set rather than learning true underlying patterns. Performance on the testing split gives an unbiased estimate of how the model will perform on new data in production.

PART 8 — Regression Modeling

Regression predicts a continuous quantitative value (such as temperature, power consumption, or pressure) across a continuous numerical domain.

```python
from sklearn.linear_model import LinearRegression

# Instantiate Linear Regression Model
reg_model = LinearRegression()

# Train Model Weights
reg_model.fit(X_train, y_train)

# Generate Out-of-Sample Predictions
y_pred_reg = reg_model.predict(X_test)
```

Mathematical Foundations
Linear regression assumes the target variable y is a linear combination of input features X:
y_hat = b0 + b1*x1 + b2*x2 + ... + bn*xn

Where:
y_hat is the predicted continuous output.
b0 is the intercept parameter.
b1, b2, ..., bn are the regression coefficients for each feature.
x1, x2, ..., xn are the input feature values.

Internal Operations of .fit() and .predict()
.fit(X_train, y_train): Computes the optimal coefficients using Ordinary Least Squares (OLS).
.predict(X_test): Multiplies the learned coefficient vector by the feature values in X_test to generate predictions.

PART 9 — Regression Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred_reg)
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
```

1. Mean Absolute Error (MAE)
Interpretation: The average absolute difference between predicted and actual values.
Units: Expressed in the same units as the target variable (e.g., °C).
Use Case: Provides a straightforward measure of average model error without over-weighting outliers.

2. Mean Squared Error (MSE)
Interpretation: The average of the squared prediction errors.
Units: Expressed in squared units of the target (e.g., °C²).
Use Case: Squaring error terms heavily penalizes larger errors, making MSE a useful metric when large deviations are particularly undesirable.

3. Coefficient of Determination (R^2)
Interpretation: The proportion of variance in the target variable explained by the model's features relative to a simple baseline mean predictor.
Values:
R^2 = 1.0: Perfect predictions; the model explains all variance.
R^2 = 0.0: Performance is equivalent to predicting the mean of the target.
R^2 < 0.0: Performance is worse than predicting the baseline mean.

PART 10 — Regression Metrics Comparison

| Metric | Sensitive to Outliers? | Units | Ideal Target Value |
|---|---|---|---|
| MAE | No (Linear scaling) | Same as target variable | 0.0 |
| MSE | Yes (Quadratic scaling) | Squared target units | 0.0 |
| R^2 | Yes | Unitless ratio | 1.0 |

Application to Smart Manufacturing
In IoT energy and thermal monitoring:
MAE provides a simple metric for operational staff (e.g., "the model is off by an average of 1.2 kW").
MSE helps flag dangerous operational spikes, as large deviations produce disproportionately large error values.
R^2 provides a high-level summary of overall model fit across varying operational conditions.

PART 11 — Binary Target Construction for Classification

Classification tasks assign samples to discrete categories rather than predicting continuous values. In this notebook, continuous sensor values are converted into binary operational flags (such as High vs. Normal state).

```python
# Thresholding continuous values into binary target states
threshold = df['temp'].median()
df['High_Temp'] = (df['temp'] > threshold).astype(int)
```

Step-by-Step Code Execution
threshold = df['temp'].median(): Computes the median temperature across all samples.
df['temp'] > threshold: Generates a Boolean Series (True where temperature exceeds the median, False otherwise).
.astype(int): Converts the Boolean values to binary integers (True -> 1, False -> 0).

```text
Continuous Target (temp)      Thresholding (> Median)     Binary Target (High_Temp)
       24.5 °C                ------------------------>             1 (High)
       19.2 °C                ------------------------>             0 (Normal)
```

This transformation sets up a binary classification problem, converting a continuous prediction task into a discrete state estimation task (e.g., predicting normal vs. elevated operational states).

PART 12 — Classification Modeling with Logistic Regression

Despite its name, Logistic Regression is a linear model designed for binary classification.

```python
from sklearn.linear_model import LogisticRegression

# Instantiate Classification Estimator
clf_model = LogisticRegression(max_iter=1000)

# Split Dataset for Classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, df['High_Temp'], test_size=0.2, random_state=42
)

# Train Classification Model
clf_model.fit(X_train_c, y_train_c)

# Generate Binary Predictions
y_pred_class = clf_model.predict(X_test_c)
```

Mathematical Foundations
Logistic Regression models the probability that a given input belongs to the positive class using the sigmoid function:
P(y=1|x) = 1 / (1 + e^-z)

Where z is the linear combination of features and weights:
z = b0 + b1*x1 + b2*x2 + ... + bn*xn

Classification Decision Rule
The continuous probability output is converted to a binary class decision using a threshold (default is 0.5):
y_hat = 1 if P(y=1|x) >= 0.5 else 0

PART 13 — Classification Evaluation Metrics

Classification performance is evaluated using metrics derived from the Confusion Matrix.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test_c, y_pred_class)
prec = precision_score(y_test_c, y_pred_class)
rec = recall_score(y_test_c, y_pred_class)
f1 = f1_score(y_test_c, y_pred_class)
```

The Confusion Matrix

| | Actual Negative (y=0) | Actual Positive (y=1) |
|---|---|---|
| Predicted Negative | True Negative (TN): Correctly predicted normal operation | False Negative (FN): Missed anomaly/high temperature |
| Predicted Positive | False Positive (FP): False alarm raised | True Positive (TP): Correctly detected high temperature |

Metric Formulas and IoT Interpretations

1. Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Meaning: The proportion of all predictions that were correct.
Limitation: Can be misleading on imbalanced datasets.

2. Precision
Precision = TP / (TP + FP)
Meaning: Out of all cases predicted as positive, how many were actually positive?
IoT Context: High precision means low false alarms.

3. Recall (Sensitivity)
Recall = TP / (TP + FN)
Meaning: Out of all actual positive cases, how many were correctly detected by the model?
IoT Context: High recall means few missed events.

4. F1-Score
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
Meaning: The harmonic mean of Precision and Recall, balancing both metrics into a single score.

PART 14 — Comparing Regression and Classification

| Dimension | Regression Paradigm | Classification Paradigm |
|---|---|---|
| Target Data Type | Continuous numerical values | Discrete categories / binary states |
| Example Output | Continuous temperature reading (24.73 °C) | Binary operational status (0 = Normal, 1 = Overheating) |
| Primary Algorithm | Linear Regression | Logistic Regression |
| Core Objective | Minimize numerical prediction error | Maximize class boundary separation |
| Evaluation Metrics | MAE, MSE, R^2 | Accuracy, Precision, Recall, F1-Score |

Why Evaluation Metrics Cannot Be Interchanged
Regression metrics on classification tasks: Computing MAE on binary labels (0 or 1) treats misclassifications as simple distance errors, failing to capture class distribution nuances.
Classification metrics on regression tasks: Accuracy cannot be applied directly to continuous values because exact continuous numerical matches almost never occur.

PART 15 — Scikit-Learn API Reference

| Library / Module | Function / Class | Purpose in Pipeline |
|---|---|---|
| pandas | read_csv() | Imports smart_manufacturing_data.csv into a DataFrame |
| pandas | DataFrame.isnull().sum() | Checks for missing (NaN) values across columns |
| sklearn.model_selection | train_test_split() | Splits features and targets into training and testing sets |
| sklearn.linear_model | LinearRegression | Fits a linear model for continuous target prediction |
| sklearn.linear_model | LogisticRegression | Fits a logistic model for binary classification |
| sklearn.metrics | mean_absolute_error | Evaluates average absolute error for regression |
| sklearn.metrics | mean_squared_error | Evaluates average squared error for regression |
| sklearn.metrics | r2_score | Computes coefficient of determination (R^2) |
| sklearn.metrics | accuracy_score | Computes proportion of correct predictions |
| sklearn.metrics | precision_score | Computes precision |
| sklearn.metrics | recall_score | Computes recall |
| sklearn.metrics | f1_score | Computes harmonic mean of precision and recall |

PART 16 — Sequential Cell Breakdown

Cell 1: Environment Ingestion and File Load
```python
import pandas as pd
import numpy as np

df = pd.read_csv('smart_manufacturing_data.csv')
```
What it does: Imports libraries and loads the CSV dataset into memory.

Cell 2: Structural Verification
```python
print(df.shape)
df.head()
```
What it does: Prints dataset dimensions and displays the top 5 records.

Cell 3: Data Integrity Audit
```python
print(df.isnull().sum())
df.info()
```
What it does: Checks for null values and displays data types for all columns.

Cell 4: Statistical Summaries
```python
df.describe()
```
What it does: Displays descriptive statistics for numeric variables.

Cell 5: Regression Feature Isolation
```python
X = df[['co', 'humidity', 'lpg', 'smoke']]
y = df['temp']
```
What it does: Separates predictor features into matrix X and continuous target into vector y.

Cell 6: Train/Test Partitioning (Regression)
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
What it does: Splits X and y into training (80%) and testing (20%) sets.

Cell 7: Model Fitting (Linear Regression)
```python
from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
```
What it does: Trains a Linear Regression model on the training set.

Cell 8: Prediction & Evaluation (Regression)
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_reg = reg_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_reg))
print("MSE:", mean_squared_error(y_test, y_pred_reg))
print("R2 Score:", r2_score(y_test, y_pred_reg))
```
What it does: Predicts values and evaluates performance using MAE, MSE, and R^2.

Cell 9: Binary Classification Target Construction
```python
threshold = df['temp'].median()
df['High_Temp'] = (df['temp'] > threshold).astype(int)
y_class = df['High_Temp']
```
What it does: Creates a binary target column indicating whether temperature is high (1) or normal (0).

Cell 10: Classification Train/Test Partitioning
```python
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
```
What it does: Splits features X and the binary target y_class into training and testing sets.

Cell 11: Logistic Regression Training & Evaluation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_c, y_train_c)
y_pred_class = clf_model.predict(X_test_c)

print("Accuracy:", accuracy_score(y_test_c, y_pred_class))
print("Precision:", precision_score(y_test_c, y_pred_class))
print("Recall:", recall_score(y_test_c, y_pred_class))
print("F1 Score:", f1_score(y_test_c, y_pred_class))
```
What it does: Trains a Logistic Regression classifier, generates predictions, and computes classification metrics.

PART 17 — Data Transformation Flow

```text
REGRESSION PIPELINE:
Raw CSV -> df -> Feature Selection (X, y) -> train_test_split() -> X_train, X_test, y_train, y_test -> LinearRegression.fit() -> predict() -> Evaluate: MAE, MSE, R^2

CLASSIFICATION PIPELINE:
Raw DataFrame -> Target Construction (High_Temp) -> Feature Selection (X, y_class) -> train_test_split() -> X_train_c, X_test_c, y_train_c, y_test_c -> LogisticRegression.fit() -> predict() -> Evaluate: Acc, Prec, Rec, F1
```

PART 18 — Critical Evaluation and Potential Improvements

Baseline Implementation Strengths
Clear Pipeline: Follows a standard ML pipeline structure.
Proper Data Splitting: Avoids training set leakage by evaluating on an independent out-of-sample test split.
Reproducibility: Sets explicit pseudo-random seeds.

Areas for Improvement
1. Feature Scaling
Current State: Input variables with different scales are passed directly to models without normalization.
Recommended Improvement: Apply standard feature scaling using StandardScaler.

2. Temporal Data Structure
Current State: Time-series telemetry data is treated as independent and identically distributed samples.
Recommended Improvement: Use time-series splitting (TimeSeriesSplit) to evaluate performance chronologically.

3. Target Construction
Current State: The classification target is derived directly from the feature temp using median thresholding, creating a potential target leak if temp remains in the input features.
Recommended Improvement: Ensure the feature used to create the target is excluded from the input feature set X.

PART 19 — High-Level Assignment Summary

"In this assignment, we build an end-to-end machine learning pipeline using Python and scikit-learn to analyze smart manufacturing IoT sensor data from smart_manufacturing_data.csv.

First, we load the raw sensor readings into a pandas.DataFrame and perform exploratory data analysis using .shape, .info(), .head(), and .describe(). This checks dataset dimensions, confirms data types, and verifies there are no missing (NaN) values across all 405,659 records.

Next, we identify numerical features (co, humidity, lpg, smoke) and set up two predictive modeling tasks:

Regression: We fit a LinearRegression model to predict continuous air temperature (temp). We evaluate performance on an independent out-of-sample test set (20% split) using Mean Absolute Error (MAE), Mean Squared Error (MSE), and the R^2 score to measure numerical prediction accuracy.

Classification: We construct a binary target (High_Temp) using median thresholding to represent elevated operational states. We train a LogisticRegression model to classify operational states and evaluate performance using Accuracy, Precision, Recall, and F1-Score derived from the Confusion Matrix.

Finally, we compare the two modeling approaches: regression predicts continuous numeric values evaluated by distance metrics, while classification predicts discrete operational categories evaluated by decision metrics."

PART 20 — Viva / Interview Questions & Answers

1. What is the overarching objective of this Jupyter Notebook?
Answer: The notebook demonstrates an end-to-end machine learning pipeline on smart manufacturing data.

2. What library is used for data manipulation, and what does read_csv do?
Answer: pandas is used for data manipulation. pd.read_csv() parses CSVs into a DataFrame.

3. What is the difference between a pandas DataFrame and a Series?
Answer: A DataFrame is a 2D tabular data structure; a Series is 1D.

4. Why do we inspect df.isnull().sum() prior to model fitting?
Answer: To check for missing values, which can cause ML algorithms to fail.

5. What is the difference between numerical and categorical variables?
Answer: Numerical variables are continuous quantities; categorical are discrete groups.

6. Why can't string columns like device MAC addresses be passed directly into Linear Regression?
Answer: Linear models require numeric inputs; strings must be encoded first.

7. What is a feature matrix (X), and what is a target vector (y)?
Answer: X contains the predictor variables; y is what the model tries to predict.

8. What is data leakage?
Answer: When information from outside the training dataset is accidentally used during training.

9. Why do we split data into training and testing sets?
Answer: To evaluate how well the model generalizes to unseen data.

10. What does the random_state parameter control?
Answer: It sets the seed for random operations, ensuring reproducibility.

11. What is the equation for Linear Regression?
Answer: y_hat = b0 + b1*x1 + ... + bn*xn

12. How does LinearRegression.fit() work?
Answer: It uses Ordinary Least Squares to minimize squared residuals.

13. What is Mean Absolute Error (MAE)?
Answer: The average absolute difference between predicted and actual values.

14. What is Mean Squared Error (MSE)?
Answer: The average squared difference, which penalizes larger errors more heavily.

15. How do you interpret an R^2 score of 0.85?
Answer: 85% of the variance in the target is explained by the features.

16. What does a negative R^2 score indicate?
Answer: The model performs worse than just predicting the mean.

17. How is a binary target created here?
Answer: By applying median thresholding (e.g., temp > median).

18. Why is Logistic Regression used for classification?
Answer: It estimates class probabilities using the sigmoid function.

19. What is the Sigmoid activation function?
Answer: A function that bounds outputs between 0 and 1.

20. How are probabilities converted into predictions?
Answer: Using a decision threshold (e.g., 0.5).

21. What is a Confusion Matrix?
Answer: A table summarizing TP, TN, FP, and FN.

22. What is a False Positive vs. False Negative?
Answer: FP is a false alarm; FN is a missed detection.

23. What is Precision?
Answer: Accuracy of positive predictions (TP / (TP + FP)).

24. What is Recall?
Answer: Proportion of actual positives detected (TP / (TP + FN)).

25. What is the F1-Score?
Answer: The harmonic mean of Precision and Recall.

26. Why can Accuracy be misleading?
Answer: On imbalanced datasets, predicting the majority class gives high accuracy but poor actual performance.

27. Why can't continuous regression tasks be evaluated using Accuracy?
Answer: Exact numerical matches almost never occur.

28. What is Feature Scaling?
Answer: Normalizing input features to a common scale.

29. How would you adjust data splitting for time-series data?
Answer: Use chronological time-series splitting to avoid look-ahead bias.

30. How can this pipeline be deployed?
Answer: Models can be saved and deployed as microservices to evaluate real-time telemetry.
