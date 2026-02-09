# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the libraries and read the data frame using pandas

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MANJUSRI KAVYA R
RegisterNumber: 212224040186
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("Salary.csv")

# Print first 5 rows
print("data.head():")
print("First 5 rows of the dataset:")
print(data.head(), "\n")

# Print info
print("data.info():")
print(data.info(), "\n")

# Check for nulls
print("data.isnull().sum():")
print("Missing Values in Each Column:")
print(data.isnull().sum(), "\n")

# Encode 'Position'
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

print("data.head() for salary")
print("Data after encoding 'Position':")
print(data.head(), "\n")

# Features and target
X = data[["Position", "Level"]]
y = data["Salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Train model
dt = DecisionTreeRegressor(random_state=2)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score (R²): {r2}\n")

# Predict salary for Position="Manager", Level=6
pos_encoded = le.transform(["Manager"])[0]
test_input = pd.DataFrame([[pos_encoded, 6]], columns=["Position", "Level"])
pred_salary = dt.predict(test_input)[0]

print(f"Predicted salary for input [[{pos_encoded}, 6]]: {pred_salary}")

```

## Output:

data.head():

<img width="299" height="146" alt="image" src="https://github.com/user-attachments/assets/e01b99ff-48fc-456f-9204-7b20357ed315" />

data.isnull().sum():

<img width="244" height="108" alt="image" src="https://github.com/user-attachments/assets/807d94de-179e-40c6-970d-70b8c02e2957" />

data.head() for salary:

<img width="251" height="144" alt="image" src="https://github.com/user-attachments/assets/4e1fcbd3-7984-429d-9bd0-1ea778b1586c" />

MSE and r2 value:

<img width="346" height="57" alt="image" src="https://github.com/user-attachments/assets/3f8907d3-e7c2-4e71-a80f-4e5f364b5649" />

data prediction:

<img width="376" height="33" alt="image" src="https://github.com/user-attachments/assets/106ce66c-461d-4baf-af0c-e3314582ace6" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
