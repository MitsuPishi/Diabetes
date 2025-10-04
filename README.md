# Diabetes Detection with Machine Learning

This repository contains code for predicting **types of diabetes** using machine learning models. The dataset used comes from the **FINAI competition on Quera**.

## Project Overview

The goal of this project is to build and evaluate different machine learning models to classify patients into one of three categories:

* **0** → No diabetes
* **1** → Type 1 diabetes
* **2** → Type 2 diabetes

We experiment with multiple models, including:

* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **CatBoost Classifier**

Feature selection is performed using **feature importances from Random Forest**, and the models are retrained on the most significant features to improve performance.

## Dataset

* **Source**: Quera – FINAI competition dataset (`train.csv`, `test.csv`)
* **Target variable**: `Diabetic` (multiclass classification: 0, 1, 2)
* **Important features**: Extracted using Random Forest feature importances

### Preprocessing Steps

1. Dropped irrelevant column: `ADM_RNO1`
2. Handled missing values:

   * Dropped rows with missing values in training data
   * Filled missing values in `Smoked` column of test data with median
3. Scaled numerical features with `StandardScaler`

## Models and Evaluation

The following models were trained and evaluated:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**
* **CatBoost**

Evaluation metric: **F1-Score (macro-averaged for multiclass)**

Example F1-scores on validation set:

```
Random Forest: 64.6%
Logistic Regression: 61.9%
XGBoost: 65.5%
CatBoost: 65.6%
```

(*Exact scores depend on dataset and run configuration*)

## Feature Importance

The top 25 most important features were extracted from the Random Forest model. Training was repeated using only these features, improving both speed and accuracy.

## Submission

For the competition, predictions were generated on the test dataset and saved in a file named `submission.csv`:

```
diabet
0
1
2
0
...
```
## Acknowledgements

* Dataset provided by **FINAI competition on Quera**
* Implemented using **scikit-learn**, **XGBoost**, and **CatBoost**

---

🚀 Future improvements may include hyperparameter tuning, ensembling models, and experimenting with deep learning architectures for multiclass classification.
