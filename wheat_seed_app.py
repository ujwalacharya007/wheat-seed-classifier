import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("wheat_seed_classifier_model.pkl")

# Function to map class label to wheat variety name
def get_variety_name(class_label):
    label_mapping = {
        1: "Kama",
        2: "Rosa",
        3: "Canadian",
        4: "Nl 297",
        5: "Vijay"
    }
    return label_mapping.get(class_label, "Unknown")

# Streamlit App Title
st.title("🌾 Wheat Seed Variety Classifier")
st.write("Enter the features of the wheat seed below:")

# Input fields for all features
area = st.number_input("Area", min_value=0.0)
perimeter = st.number_input("Perimeter", min_value=0.0)
compactness = st.number_input("Compactness", min_value=0.0)
length = st.number_input("Length of Kernel", min_value=0.0)
width = st.number_input("Width of Kernel", min_value=0.0)
asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0)
groove_length = st.number_input("Length of Kernel Groove", min_value=0.0)

# Predict button
if st.button("Predict Variety"):
    input_data = np.array([[area, perimeter, compactness, length, width, asymmetry, groove_length]])
    prediction = model.predict(input_data)[0]
    variety = get_variety_name(prediction)
    st.success(f"🌟 Predicted Variety: **{variety}**")
