import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('insurance_model.pkl')

# 2. Set up the page title and description
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details below to get an estimated insurance cost.")

# 3. Create the Input Form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

with col2:
    children = st.slider("Number of Children", 0, 10, 0)
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    smoker = st.radio("Are you a smoker?", ["yes", "no"])

# 4. The Prediction Logic
if st.button("Predict Cost"):
    # Create a dataframe matching the model's training input exactly
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Predict
    prediction = model.predict(user_data)
    
    # Show result
    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")