This project focuses on the **SECOM (Semiconductor Manufacturing) Dataset**, dealing with high-dimensional sensor data to predict process failures.

---

# SECOM Sensor Data Analysis & Failure Prediction

This repository contains a comprehensive data science pipeline for analyzing semiconductor manufacturing process data. The goal is to handle a highly imbalanced dataset and high-dimensional sensor readings to accurately predict "Fail" (1) vs "Pass" (-1) outcomes.

## 📌 Project Overview
In semiconductor manufacturing, monitoring sensors is crucial for quality control. This project explores the **SECOM dataset** from the UCI Machine Learning Repository, which consists of 1,567 examples, each with 591 features (sensor readings).

## 📂 File Structure
* **`uci_sensor_data1.ipynb`**: The primary research notebook containing data cleaning, exploratory analysis, and model experimentation.
* **`uci_sensor_data1.py`**: The production-ready Python script converted from the notebook for easier deployment and batch processing.

## 🛠️ Technical Workflow

### 1. Data Cleaning & Preprocessing
* **Handling Missing Values**: Identification of sensors with high null-value percentages. Sensors with excessive missing data are dropped, while others are imputed (Median/Mean).
* **Constant Feature Removal**: Dropping sensors that show zero variance (constants), as they provide no predictive power.
* **Imbalance Handling**: The dataset is heavily skewed towards "Pass" results. The project implements techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the classes before training.

### 2. Feature Selection & Dimensionality Reduction
Given the 500+ features, the project uses:
* **Correlation Analysis**: To remove highly redundant sensors.
* **PCA (Principal Component Analysis)**: Reducing dimensionality while retaining maximum variance to improve model efficiency and reduce noise.

### 3. Statistical Analysis
* **VIF (Variance Inflation Factor)**: Used to detect multicollinearity among sensor readings.
* **Visual Analysis**: Utilizing histograms and boxplots to identify sensor drifts and outliers.

### 4. Machine Learning Models
The project evaluates several classifiers to find the best fit for high-dimensional sensor data:
* **Logistic Regression** (Baseline)
* **Random Forest Classifier**
* **XGBoost / LightGBM** (Optimized for performance on imbalanced data)
* **Support Vector Machines (SVM)**

## 🚀 Getting Started

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

### Usage
1.  Clone the repository:
    ```bash
    git clone https://github.com/gokilanr/Secom.git
    ```
2.  Run the notebook or the python script:
    ```bash
    python uci_sensor_data1.py
    ```

## 📊 Results
The final model focuses on maximizing **Recall** and **F1-Score** rather than just Accuracy, ensuring that potential manufacturing failures are not missed (minimizing False Negatives).

