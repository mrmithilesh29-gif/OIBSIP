import streamlit as st
import numpy as np
import joblib

# Load trained Random Forest model and scaler
best_rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")   # save your scaler during training with joblib.dump(scaler,"scaler.pkl")

st.title("🚗 Car Price Prediction (Random Forest)")

# Input form
year = st.number_input("Year of Purchase", min_value=2000, max_value=2026, step=1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
driven_kms = st.number_input("Driven Kms", min_value=0, step=100)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.number_input("Number of Previous Owners", min_value=0, step=1)
car_age = st.number_input("Car Age (years)", min_value=0, step=1)

# Encode categorical values
fuel_map = {"CNG":0, "Diesel":1, "Petrol":2}
seller_map = {"Individual":0, "Dealer":1}
trans_map = {"Manual":0, "Automatic":1}

fuel_type_val = fuel_map[fuel_type]
seller_type_val = seller_map[seller_type]
transmission_val = trans_map[transmission]

# Prepare input with 8 features in the same order as training
input_data = np.array([[year, present_price, driven_kms, fuel_type_val,
                        seller_type_val, transmission_val, owner, car_age]])

# Apply the same scaling as training
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Selling Price"):
    prediction = best_rf.predict(input_data_scaled)[0]
    st.success(f"Predicted Selling Price: {prediction:.2f} lakhs")
