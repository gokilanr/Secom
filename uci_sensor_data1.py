#!/usr/bin/env python
# coding: utf-8

# **About Dataset**
# Context
# Manufacturing process feature selection and categorization
# 
# Content
# Abstract: Data from a semi-conductor manufacturing process
# 
# Data Set Characteristics: Multivariate
# Number of Instances: 1567
# Area: Computer
# Attribute Characteristics: Real
# Number of Attributes: 591
# Date Donated: 2008-11-19
# Associated Tasks: Classification, Causal-Discovery
# Missing Values? Yes
# A complex modern semi-conductor manufacturing process is normally under consistent
# surveillance via the monitoring of signals/variables collected from sensors and or
# process measurement points. However, not all of these signals are equally valuable
# in a specific monitoring system. The measured signals contain a combination of
# useful information, irrelevant information as well as noise. It is often the case
# that useful information is buried in the latter two. Engineers typically have a
# much larger number of signals than are actually required. If we consider each type
# of signal as a feature, then feature selection may be applied to identify the most
# relevant signals. The Process Engineers may then use these signals to determine key
# factors contributing to yield excursions downstream in the process. This will
# enable an increase in process throughput, decreased time to learning and reduce the
# per unit production costs.
# 
# To enhance current business improvement techniques the application of feature
# selection as an intelligent systems technique is being investigated.
# 
# The dataset presented in this case represents a selection of such features where
# each example represents a single production entity with associated measured
# features and the labels represent a simple pass/fail yield for in house line
# testing, figure 2, and associated date time stamp. Where .1 corresponds to a pass
# and 1 corresponds to a fail and the data time stamp is for that specific test
# point.
# 
# Using feature selection techniques it is desired to rank features according to
# their impact on the overall yield for the product, causal relationships may also be
# considered with a view to identifying the key features.
# 
# Results may be submitted in terms of feature relevance for predictability using
# error rates as our evaluation metrics. It is suggested that cross validation be
# applied to generate these results. Some baseline results are shown below for basic
# feature selection techniques using a simple kernel ridge classifier and 10 fold
# cross validation.
# 
# Baseline Results: Pre-processing objects were applied to the dataset simply to
# standardize the data and remove the constant features and then a number of
# different feature selection objects selecting 40 highest ranked features were
# applied with a simple classifier to achieve some initial results. 10 fold cross
# validation was used and the balanced error rate (*BER) generated as our initial
# performance metric to help investigate this dataset.
# 
# SECOM Dataset: 1567 examples 591 features, 104 fails
# 
# FSmethod (40 features) BER % True + % True - %
# S2N (signal to noise) 34.5 +-2.6 57.8 +-5.3 73.1 +2.1
# Ttest 33.7 +-2.1 59.6 +-4.7 73.0 +-1.8
# Relief 40.1 +-2.8 48.3 +-5.9 71.6 +-3.2
# Pearson 34.1 +-2.0 57.4 +-4.3 74.4 +-4.9
# Ftest 33.5 +-2.2 59.1 +-4.8 73.8 +-1.8
# Gram Schmidt 35.6 +-2.4 51.2 +-11.8 77.5 +-2.3
# 
# Attribute Information:
# 
# Key facts: Data Structure: The data consists of 2 files the dataset file SECOM
# consisting of 1567 examples each with 591 features a 1567 x 591 matrix and a labels
# file containing the classifications and date time stamp for each example.
# 
# As with any real life data situations this data contains null values varying in
# intensity depending on the individuals features. This needs to be taken into
# consideration when investigating the data either through pre-processing or within
# the technique applied.
# 
# The data is represented in a raw text file each line representing an individual
# example and the features seperated by spaces. The null values are represented by
# the 'NaN' value as per MatLab.

# In[1]:


get_ipython().system('pip install scikit-optimize')


# In[2]:


# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV
import matplotlib.pyplot as plt 


# In[3]:


#  Load Data
data = pd.read_csv("uci-secom.csv")
labels = data['Pass/Fail']
features = data.drop(columns=['Pass/Fail', 'Time'])  # Exclude non-sensor data like timestamp



# In[11]:


#missing values 
data.isnull().sum()


# In[4]:


# Handle Missing Values
# Impute missing values (NaNs) using the mean of each column
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)



# In[5]:


# Standardize the Data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)



# In[6]:


# Feature Selection
# Select top 40 features using ANOVA F-test
selector = SelectKBest(f_classif, k=40)
features_selected = selector.fit_transform(features_scaled, labels)



# In[7]:


#  Split Data
X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)


# In[8]:


from sklearn.preprocessing import StandardScaler

# Assuming X_train is your training data
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Then you can use it to transform both the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 


# In[10]:


import joblib

# Save the fitted scaler
joblib.dump(scaler, 'scaler_filename.pkl')


# In[52]:


# Create and train the LightGBM model
lgb_model = LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)

# Evaluation
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgb))


# In[53]:


# LightGBM with Bayesian Optimization
lgbm_model = LGBMClassifier()

# Define the parameter search space for LightGBM
param_space_lgbm = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'n_estimators': (50, 500),
    'subsample': (0.5, 1.0)
}

# Use Bayesian optimization for hyperparameter tuning with LightGBM
bayes_cv_lgbm = BayesSearchCV(estimator=lgbm_model, search_spaces=param_space_lgbm, cv=10, n_iter=32, n_jobs=-1, random_state=42)

# Fit the LightGBM model
bayes_cv_lgbm.fit(X_train, y_train)

# Evaluate the LightGBM model
y_pred_lgbm = bayes_cv_lgbm.best_estimator_.predict(X_test)
ber_lgbm = balanced_accuracy_score(y_test, y_pred_lgbm)

print(f"Balanced Error Rate (LightGBM): {1 - ber_lgbm:.4f}")
print(f"Best Parameters (LightGBM): {bayes_cv_lgbm.best_params_}")



# In[55]:


# Create the LightGBM model with the best parameters
best_params = {
    'learning_rate': 0.010486266869304801,
    'max_depth': 3,
    'n_estimators': 64,
    'subsample': 0.529750181575546
}

# Initialize and train the LightGBM model
hy_lgb_model = LGBMClassifier(**best_params)
hy_lgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = hy_lgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[56]:


# Evaluation
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgb))


# In[57]:


# Predict probabilities for ROC AUC calculation
y_pred_lgb_proba = hy_lgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC for XGBoost
roc_auc_lgb = roc_auc_score(y_test, y_pred_lgb_proba)


# In[58]:


# Create and train the XGBoost model
xgb_model = XGBClassifier()
# Change class labels from -1 and 1 to 0 and 1
# Use astype(int) to ensure the result is integer
y_train = ((y_train + 1) / 2).astype(int)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
#Remap the predicted values back to -1 and 1
y_pred_xgb = (y_pred_xgb * 2) -1

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[59]:


#  Model Building - XGBoost with Bayesian Optimization
xgb_model = XGBClassifier()

# Define the parameter search space
param_space = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'n_estimators': (50, 500),
    'subsample': (0.5, 1.0)
}

# Use Bayesian optimization for hyperparameter tuning
bayes_cv = BayesSearchCV(estimator=xgb_model, search_spaces=param_space, cv=10, n_iter=32, n_jobs=-1, random_state=42)

# Ensure labels are 0 and 1
y_train = y_train.replace(-1, 0)

# Fit the model
bayes_cv.fit(X_train, y_train)

# Evaluate the model
y_pred = bayes_cv.best_estimator_.predict(X_test)
ber = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Error Rate: {1 - ber:.4f}")
print(f"Best Parameters: {bayes_cv.best_params_}")


# In[60]:


# Define the best parameters
best_params = {
    'learning_rate': 0.010001895862575045,
    'max_depth': 8,
    'n_estimators': 71,
    'subsample': 0.9494077575579702,
    'objective': 'binary:logistic'
}

# Initialize and train the XGBoost model
xgb_model_hy = XGBClassifier(**best_params)
xgb_model_hy.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model_hy.predict(X_test)

# Calculate the Balanced Error Rate (BER)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Error Rate (BER): {ber:.4f}")
# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[61]:


# Predict probabilities for ROC AUC calculation
y_pred_xgb_proba = xgb_model_hy.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC for XGBoost
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)


# In[62]:


# Train SVM for classification
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# Predicting with SVM
y_pred_svm = svm_model.predict(X_test)

# Evaluation metrics for SVM
print("\n--- SVM Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='macro'))
print("Recall:", recall_score(y_test, y_pred_svm, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_svm, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))


# In[63]:


# Train the model
isolation_forest = IsolationForest(contamination='auto', random_state=42)
isolation_forest.fit(X_train)

# Predicting anomalies
y_pred_if = isolation_forest.predict(X_test)

# Convert IsolationForest output (1 -> normal, -1 -> anomaly) to match target (1 -> pass, -1 -> fail)
y_pred_if[y_pred_if == 1] = 1  # Normal = 1
y_pred_if[y_pred_if == -1] = -1  # Anomaly = -1

# Evaluation metrics for Isolation Forest
print("\n--- Isolation Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_if))
print("Precision:", precision_score(y_test, y_pred_if, average='macro'))
print("Recall:", recall_score(y_test, y_pred_if, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_if, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_if))


# In[64]:


# Plotting the ROC curve for all models
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_lgb_proba)


plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {roc_auc_lgb:.2f})')


# Plot the random chance line
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='best')
plt.show()

# Print AUC scores
print(f"XGBoost ROC-AUC: {roc_auc_xgb:.4f}")
print(f"LightGBM ROC-AUC: {roc_auc_lgb:.4f}")


# The ROC curve comparison you shared shows the performance of two models: XGBoost and LightGBM. Based on the graph, here are the observations:
# 
# XGBoost has an AUC of 0.75, indicating better performance compared to LightGBM for this particular dataset.
# LightGBM has an AUC of 0.71, which is slightly lower than XGBoost.
# The random chance line (AUC = 0.50) serves as a baseline, showing that both models are performing significantly better than random classification.

# model  save in pickle file

# In[68]:


# Save using joblib
import joblib
joblib.dump(xgb_model_hy, 'xgboost_model.pkl')
# Save the fitted scaler
joblib.dump(scaler, 'scaler_filename.pkl')


# In[70]:


# Load the saved scaler and model
scaler = joblib.load('scaler_filename.pkl')
model = joblib.load('xgboost_model.pkl')

# Now you can transform the input data using the loaded scaler
input_df_scaled = scaler.transform(X_train_scaled)

# Predict using the loaded model
prediction = model.predict(X_train_scaled)


# In[13]:


pip install imblearn


# In[14]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix


# In[16]:


# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Change class labels from -1 and 1 to 0 and 1 for y_resampled
# Use astype(int) to ensure the result is integer
y_resampled = ((y_resampled + 1) / 2).astype(int)

# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_resampled, y_resampled)

# Make predictions
y_pred = model.predict(X_test)
#Remap the predicted values back to -1 and 1
y_pred_xgb = (y_pred * 2) -1

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")


# In[17]:


#  Model Building - XGBoost with Bayesian Optimization
xgb_model = XGBClassifier()

# Define the parameter search space
param_space = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'n_estimators': (50, 500),
    'subsample': (0.5, 1.0)
}

# Use Bayesian optimization for hyperparameter tuning
bayes_cv = BayesSearchCV(estimator=xgb_model, search_spaces=param_space, cv=10, n_iter=32, n_jobs=-1, random_state=42)

# Ensure labels are 0 and 1
y_train = y_train.replace(-1, 0)

# Fit the model
bayes_cv.fit(X_train, y_train)

# Evaluate the model
y_pred = bayes_cv.best_estimator_.predict(X_test)
ber = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Error Rate: {1 - ber:.4f}")
print(f"Best Parameters: {bayes_cv.best_params_}")




# In[18]:


# Define the best parameters
best_params = {
    'learning_rate': 0.010001895862575045,
    'max_depth': 8,
    'n_estimators': 71,
    'subsample': 0.9494077575579702,
    'objective': 'binary:logistic'
}

# Initialize and train the XGBoost model
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Balanced Error Rate (BER)
ber = 1 - balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Error Rate (BER): {ber:.4f}")


# In[19]:


print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred))


# # outliers

# In[22]:


# Required Libraries for Data Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("uci-secom.csv")  # Replace with your dataset path

# General Information
print("Data Shape:", data.shape)
print("\nData Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())



# In[23]:


# Check Missing Values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Visualize Missing Values
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()



# In[24]:


#  Distribution of Pass/Fail
plt.figure(figsize=(8, 6))
sns.countplot(x='Pass/Fail', data=data)
plt.title('Pass/Fail Distribution')
plt.show()



# In[25]:


#  Correlation Matrix (Exclude non-numeric columns like 'Time' and 'Pass/Fail')
numeric_data = data.drop(columns=['Time', 'Pass/Fail'])  # Remove non-numeric columns

# Now compute the correlation matrix for numeric data only
correlation_matrix = numeric_data.corr()

# Visualize Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[26]:


# Boxplot of Sensor Data (for outliers)
plt.figure(figsize=(15, 6))
sns.boxplot(data=data.drop(columns=['Pass/Fail', 'Time']))
plt.xticks(rotation=90)
plt.title('Boxplot of Sensor Features')
plt.show()



# In[27]:


import matplotlib.pyplot as plt
data1=data.drop(columns=['Pass/Fail', 'Time'])
# Choose 5 columns from your dataset for histograms
columns_to_plot = data1.columns[:5]  # Adjust if you want specific columns

# Plot histograms for the selected columns
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)  # Create a 2x3 grid for up to 6 plots
    data[column].hist(bins=30, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[28]:


# Function to identify outliers
def find_outliers_iqr(df):
    outlier_columns = {}

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Check only numerical columns
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            if not outliers.empty:
                outlier_columns[column] = outliers

    return outlier_columns

# Identify outlier columns
outlier_columns = find_outliers_iqr(data)

# Display outlier columns and their counts
for column, outliers in outlier_columns.items():
    print(f"Column: {column}, Outliers Count: {len(outliers)}")


# In[29]:


# Removing outliers
for column in outlier_columns.keys():
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


# In[30]:


labels = data['Pass/Fail']
features = data.drop(columns=['Pass/Fail', 'Time'])


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[33]:


# Create and train the XGBoost model
xgb_model = XGBClassifier()
# Change class labels from -1 and 1 to 0 and 1
# Use astype(int) to ensure the result is integer
y_train = ((y_train + 1) / 2).astype(int)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
#Remap the predicted values back to -1 and 1
y_pred_xgb = (y_pred_xgb * 2) -1

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# ###third method
# Time series convert into month, day, week and hour. Used PCA to reduce dimensonality

# In[34]:


#dataset load using pandas
df = pd.read_csv('uci-secom.csv')


# In[35]:


df.isnull().sum()


# In[36]:


# Select only float64 columns to fit the imputer
float64_cols = df.select_dtypes(include=['float64']).columns
df[float64_cols] = imputer.fit_transform(df[float64_cols])

print(df)


# In[39]:


import pandas as pd

# Convert the Time column to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')


# In[40]:


# Extract useful time components
df['Year'] = df['Time'].dt.year
df['Month'] = df['Time'].dt.month
df['Day'] = df['Time'].dt.day
df['Hour'] = df['Time'].dt.hour
df['Minute'] = df['Time'].dt.minute
df['Second'] = df['Time'].dt.second
df['DayOfWeek'] = df['Time'].dt.dayofweek  # Monday=0, Sunday=6


# In[41]:


#df.set_index('Time', inplace=True)


# In[42]:


df.head(5)


# In[43]:


df.columns


# In[44]:


import pandas as pd

# Assuming 'Time' column is in string format, convert it to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')


# In[45]:


# Calculate the difference between consecutive time points
df['Time_Diff'] = df['Time'].diff()

# Display the time differences
print(df[['Time', 'Time_Diff']].head(10))  # Display first 10 rows


# In[46]:


import matplotlib.pyplot as plt

# Plot the time differences
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Time_Diff'].dt.total_seconds(), marker='o')
plt.ylabel('Time Difference (seconds)')
plt.xlabel('Time')
plt.title('Time Differences Between Consecutive Timestamps')
plt.show()


# In[47]:


# Set 'Time' as the index if it's not already
df.set_index('Time', inplace=True)






# In[48]:


# Check for duplicate values in the index
duplicate_index = df.index.duplicated()

# If duplicates exist, decide how to handle them
if duplicate_index.any():
    #  Remove duplicates
    df = df[~duplicate_index]



# Resample to hourly intervals and forward fill missing values
df_resampled = df.resample('H').ffill()



# In[49]:


df.head()


# In[50]:


# Convert Time_Diff to total seconds and use it as a feature
df['Time_Diff_seconds'] = df['Time_Diff'].dt.total_seconds()


# In[51]:


# Example of forward filling missing values after resampling
df_resampled.ffill(inplace=True)


# In[52]:


# Get rows with any NaN values
nan_rows = df[df.isna().any(axis=1)]

print("Rows with any NaN values:")
print(nan_rows)

# Get specific columns that contain NaN values
nan_columns = df.columns[df.isna().any()].tolist()

print("\nColumns with NaN values:")
print(nan_columns)



# In[53]:


# Replace NaN values with 0 in the 'Time_Diff_seconds' column
df['Time_Diff_seconds'] = df['Time_Diff_seconds'].fillna(0)

print("\nDataFrame after replacing NaN with 0 in 'Time_Diff_seconds':")
print(df)


# In[54]:


# Replace NaT with 0 in the 'time_diff' column
df['Time_Diff'] = df['Time_Diff'].fillna(pd.Timestamp(0))

print("\nDataFrame after replacing NaT with 0 in 'time_diff':")
print(df)


# In[55]:


df.isnull().sum()


# In[56]:


df


# In[57]:


df.info()


# ##regular and irregular resmapled

# In[58]:


print(df_resampled.head())


# In[59]:


# Remove the first row (index 0)
df_resampled = df_resampled.drop(index='2008-01-08 02:00:00').reset_index(drop=True)



# In[60]:


df_resampled.isnull().sum()


# In[61]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[62]:


# Separate features and target
X = df.drop(columns=['Pass/Fail'])  # Features
y = df['Pass/Fail']  # Target

# Identify numeric features
numeric_features = X.select_dtypes(include=['float64', 'int']).columns

# Normalize or standardize numeric features
# For normalization
scaler = MinMaxScaler()
X_normalized = X.copy()
X_normalized[numeric_features] = scaler.fit_transform(X[numeric_features])

# Combine processed data back with the target

processed_data = pd.concat([X_normalized, y], axis=1)


print(processed_data.head())


# In[71]:


from sklearn.decomposition import PCA 

# Separate features and target
X = df.drop(columns=['Pass/Fail'])  # Features
y = df['Pass/Fail']  # Target

# Normalize numeric features
scaler = MinMaxScaler()
X_normalized = X.copy()
X_normalized[X.select_dtypes(include=['float64', 'int']).columns] = scaler.fit_transform(X.select_dtypes(include=['float64', 'int']))

# Keep only numeric features for PCA
X_processed = X_normalized.select_dtypes(include=['float64', 'int'])

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_processed)

# Create a new DataFrame with PCA features
pca_columns = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Combine PCA features with the target variable
final_data = pd.concat([X_pca_df, y.reset_index(drop=True)], axis=1)

print(final_data.head())

# Optional: Check the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)


# In[72]:


cumulative_variance = pca.explained_variance_ratio_.cumsum()
print(cumulative_variance)


# In[73]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid()
plt.show()


# In[74]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, df['Pass/Fail'], test_size=0.2, random_state=42)

# Train a model (example with Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import xgboost as xgb


# ##model

# In[76]:


# Create and train the LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)

# Evaluation
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgb))


# In[77]:


# Create and train the XGBoost model
xgb_model = XGBClassifier()
# Change class labels from -1 and 1 to 0 and 1
y_train = (y_train + 1) / 2
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
#Remap the predicted values back to -1 and 1
y_pred_xgb = (y_pred_xgb * 2) -1

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[ ]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('xgboost_model.pkl')

# Load or create a StandardScaler for consistent preprocessing (if used)
scaler = StandardScaler()

# Define a function to make predictions

def predict_anomaly(input_data):
    input_df = pd.DataFrame(input_data, index=[0])

    # Load the saved scaler and model
    scaler = joblib.load('scaler_filename.pkl')
    model = joblib.load('xgboost_model.pkl')

    # Transform the input data using the fitted scaler
    input_df_scaled = scaler.transform(input_df)

    # Predict anomaly using the loaded model
    prediction = model.predict(input_df_scaled)
    
    return prediction


# Streamlit App Interface
st.title("Anomaly Detection Web App")

st.write("""
### Input your sensor data to check for anomalies
""")

# Define input fields for features based on your dataset
input_feature_1 = st.number_input('Feature 1')
input_feature_2 = st.number_input('Feature 2')
input_feature_3 = st.number_input('Feature 3')
input_feature_4 = st.number_input('Feature 4')
input_feature_5 = st.number_input('Feature 5')
input_feature_6 = st.number_input('Feature 6')
input_feature_7 = st.number_input('Feature 7')
input_feature_8 = st.number_input('Feature 8')
input_feature_9 = st.number_input('Feature 9')
input_feature_10 = st.number_input('Feature 10')
input_feature_11 = st.number_input('Feature 11')
input_feature_12 = st.number_input('Feature 12')
input_feature_13 = st.number_input('Feature 13')
input_feature_14 = st.number_input('Feature 14')
input_feature_15 = st.number_input('Feature 15')
input_feature_16 = st.number_input('Feature 16')
input_feature_17 = st.number_input('Feature 17')
input_feature_18 = st.number_input('Feature 18')
input_feature_19 = st.number_input('Feature 19')
input_feature_20 = st.number_input('Feature 20')
input_feature_21 = st.number_input('Feature 21')
input_feature_22 = st.number_input('Feature 22')
input_feature_23 = st.number_input('Feature 23')
input_feature_24 = st.number_input('Feature 24')
input_feature_25 = st.number_input('Feature 25')
input_feature_26 = st.number_input('Feature 26')
input_feature_27 = st.number_input('Feature 27')
input_feature_28 = st.number_input('Feature 28')
input_feature_29 = st.number_input('Feature 29')
input_feature_30 = st.number_input('Feature 30')
input_feature_31 = st.number_input('Feature 31')
input_feature_32 = st.number_input('Feature 32')
input_feature_33 = st.number_input('Feature 33')
input_feature_34 = st.number_input('Feature 34')
input_feature_35 = st.number_input('Feature 35')
input_feature_36 = st.number_input('Feature 36')
input_feature_37 = st.number_input('Feature 37')
input_feature_38 = st.number_input('Feature 38')
input_feature_39 = st.number_input('Feature 39')
input_feature_40 = st.number_input('Feature 40')

# Add more input fields as per your dataset's features

# When the user clicks "Predict"
if st.button('Predict'):
    # Collect the input data
    input_data = {        
        
        'feature_1': input_feature_1,
        'feature_2': input_feature_2,
        'feature_3': input_feature_3,
        'feature_4': input_feature_4,
        'feature_5': input_feature_5,
        'feature_6': input_feature_6,
        'feature_7': input_feature_7,
        'feature_8': input_feature_8,
        'feature_9': input_feature_9,
        'feature_10': input_feature_10,
        'feature_11': input_feature_11,
        'feature_12': input_feature_12,
        'feature_13': input_feature_13,
        'feature_14': input_feature_14,
        'feature_15': input_feature_15,
        'feature_16': input_feature_16,
        'feature_17': input_feature_17,
        'feature_18': input_feature_18,
        'feature_19': input_feature_19,
        'feature_20': input_feature_20,
        'feature_21': input_feature_21,
        'feature_22': input_feature_22,
        'feature_23': input_feature_23,
        'feature_24': input_feature_24,
        'feature_25': input_feature_25,
        'feature_26': input_feature_26,
        'feature_27': input_feature_27,
        'feature_28': input_feature_28,
        'feature_29': input_feature_29,
        'feature_30': input_feature_30,
        'feature_31': input_feature_31,
        'feature_32': input_feature_32,
        'feature_33': input_feature_33,
        'feature_34': input_feature_34,
        'feature_35': input_feature_35,
        'feature_36': input_feature_36,
        'feature_37': input_feature_37,
        'feature_38': input_feature_38,
        'feature_39': input_feature_39,
        'feature_40': input_feature_40,
         # Add more features here
    }
    
    # Make a prediction using the model
    prediction = predict_anomaly(input_data)
    
    # Display the result
    if prediction == -1:
        st.error("Anomaly Detected!")
    else:
        st.success("No Anomaly Detected.")



# In[ ]:




