import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model, scaler, and label encoder
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
label_encoder = joblib.load("iris_label_encoder.pkl")

st.title("🌸 Iris Flower Classification App")

st.sidebar.header("Input Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Prepare input
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
species_name = label_encoder.inverse_transform([prediction])[0]

st.subheader("🔮 Prediction Result")
st.write(f"🌼 Predicted Species: **{species_name}**")

# Show probabilities only if model supports it
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_scaled)[0]
    st.write("Prediction Probabilities:")
    st.bar_chart(pd.DataFrame(probabilities, index=label_encoder.classes_, columns=["Probability"]))
else:
    st.info("⚠️ This model does not support probability estimates.")
