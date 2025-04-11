import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load pre-trained model and transformers
with open('cat_to_num.pkl', 'rb') as file:
    one_hot = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = load_model('model.h5')

# App title
st.title('Customer Churn Prediction using ANN')

# Sample data just to get category values (like Geography choices)
data = pd.read_csv('Churn_Modelling.csv')

# Input fields
geography = st.selectbox('Geography', one_hot.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=12, max_value=92)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_crd = st.selectbox('Has Credit Card', [0, 1])
credit_score = st.number_input('Credit Score')
balance = st.number_input('Balance')
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare input data
gender_encoded = 0 if gender == 'Male' else 1
geography_encoded = one_hot.transform([[geography]])

# Convert sparse to dense if needed
geography_encoded_df = pd.DataFrame(
    geography_encoded, 
    columns=one_hot.get_feature_names_out(['Geography'])
)

# Assemble full input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_crd],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Merge one-hot encoded columns
input_data = pd.concat([input_data, geography_encoded_df], axis=1)

# Scale the data
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input)
churn_probability = prediction[0][0]

# Output
st.subheader(f"Churn Probability: {churn_probability:.2f}")
if churn_probability > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is likely to stay.')