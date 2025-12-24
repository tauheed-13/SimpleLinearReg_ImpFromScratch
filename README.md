# SimpleLinearReg_ImpFromScratch

## Overview
This repository contains a **from-scratch implementation of Simple Linear Regression** using the **Gradient Descent algorithm** to learn the **weight** and **bias** parameters.  
No machine learning libraries (such as `scikit-learn`) are used for training. The main goal of this project is to understand the **core mathematics and optimization process** behind linear regression.

---

## Simple Linear Regression

Simple Linear Regression models the relationship between **one input feature (X)** and **one output variable (y)** using a straight line.

### Mathematical Equation
y_pred = w * X + b
Where:
- **w** → weight (slope)
- **b** → bias (intercept)
- **X** → input feature
- **y_pred** → predicted output

---

## Loss Functions

Loss functions measure how well the model’s predictions match the actual values.
MSE = (1 / n) * Σ (y - y_pred)²
MAE = (1 / n) * Σ |y - y_pred|


The objective is to **minimize the loss** by updating the weight and bias values.

---

## Gradient Descent Algorithm

Gradient Descent is an optimization technique used to minimize the loss function by iteratively updating the model parameters.

### Training Steps

1. Initialize weight and bias with random values  
2. Compute predictions using
y_pred = w * X + b
4. Calculate the loss (MSE or MAE)  
5. Compute gradients:
- Partial derivative of loss with respect to **weight**
- Partial derivative of loss with respect to **bias**
5. Update parameters:
w = w - α * (∂L / ∂w)
b = b - α * (∂L / ∂b)
where **α** is the learning rate
6. Repeat the process until the loss converges (changes very little)

---

## Types of Gradient Descent

This project explains the three main types of Gradient Descent:

### Batch Gradient Descent
- Uses the entire dataset to compute gradients  
- Stable but computationally expensive for large datasets  

### Stochastic Gradient Descent (SGD)
- Updates parameters using one data point at a time  
- Faster but introduces more noise  

### Mini-batch Gradient Descent
- Uses small batches of data  
- Provides a balance between speed and stability  

---

## Model Training Loop

- Predict output values  
- Calculate loss  
- Update weight and bias  
- Repeat until convergence  
- Use the final parameters to predict unseen data  

---

## Technologies Used

- Python  
- NumPy  
- Matplotlib  

---

## Learning Objectives

- Understand Simple Linear Regression mathematically  
- Learn how Gradient Descent works internally  
- Implement machine learning algorithms from scratch  
- Strengthen core machine learning fundamentals  

---

## Future Improvements

- Implement Mini-batch Gradient Descent  
- Compare results with `scikit-learn`  
- Extend to Multiple Linear Regression  
- Add Regularization techniques (L1 / L2)  

---

## Contribution

Contributions are welcome!  
Feel free to fork this repository, raise issues, or submit pull requests.

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub!

