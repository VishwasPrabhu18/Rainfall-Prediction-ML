import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the saved KNN model
with open('../knn_model.pkl', 'rb') as file:
    knn_loaded = pickle.load(file)

# Load the saved StandardScaler
with open('../scalerKNN.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Define the class labels
class_labels = ['No Rain', 'Rain']

# Streamlit App
st.title('Weather Prediction with K-Nearest Neighbors (KNN)')

# Input fields for relevant weather features
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
            'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']

# Number of columns per row
columns_per_row = 4

# Calculate number of rows needed
num_features = len(features)
num_rows = (num_features + columns_per_row - 1) // columns_per_row

# Arrange input fields in rows of columns_per_row inputs each
input_data = {}
for row in range(num_rows):
    row_columns = st.columns(columns_per_row)
    for col_index in range(columns_per_row):
        feature_index = row * columns_per_row + col_index
        if feature_index < num_features:
            input_data[features[feature_index]] = row_columns[col_index].number_input(
                f'{features[feature_index]}', value=0.0
            )

# Predict button
if st.button('Predict'):
    # Prepare input data as a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the loaded scaler
    input_scaled = scaler_loaded.transform(input_df)

    # Predict using the loaded KNN model
    predicted_class = knn_loaded.predict(input_scaled)[0]
    predicted_prob = knn_loaded.predict_proba(input_scaled)[0]

    # Display prediction result with custom styling
    st.subheader('Prediction Result:')
    result_text = f"Predicted Class: {class_labels[predicted_class]}"
    result_color = "green" if predicted_class == 0 else "red"
    st.markdown(f'<p style="font-size: 18px; color: {result_color};">{result_text}</p>', unsafe_allow_html=True)
    st.write(f"Probability (No Rain): {predicted_prob[0]:.2f}")
    st.write(f"Probability (Rain): {predicted_prob[1]:.2f}")
    