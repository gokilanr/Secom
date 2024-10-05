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


