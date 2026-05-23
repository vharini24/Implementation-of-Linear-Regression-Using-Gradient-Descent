# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: V HARINI
RegisterNumber:  212225040113
*/


import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("50_Startups.csv")

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Features & target
X = data.drop("Profit", axis=1)
y = data["Profit"].values

# Store feature names (VERY IMPORTANT)
features = X.columns

X = X.values.astype(float)

# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias
X = np.c_[np.ones(X.shape[0]), X]

# Train
w = np.zeros(X.shape[1])
lr = 0.01
epochs = 1000

for _ in range(epochs):
    w -= lr * (X.T.dot(X.dot(w) - y)) / len(y)

print("Weights:", w)



sample = pd.DataFrame([{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 300000,
    "State_Florida": 0,
    "State_New York": 1   # adjust based on dataset columns
}])

# Align columns automatically (VERY IMPORTANT LINE)
sample = sample.reindex(columns=features, fill_value=0)

# Convert to numpy
sample = sample.values.astype(float)

# Normalize using training stats
sample = (sample - X_mean) / X_std

# Add bias
sample = np.insert(sample, 0, 1)

print("Predicted Profit:", np.dot(sample, w))
```

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
