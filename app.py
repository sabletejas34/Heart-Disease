import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_target(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred,prob

# Streamlit UI components
st.title("Heart Disease Prediction")

# Input fields for each parameter
age = st.number_input("age", min_value=29, max_value=80, value=29, step=1)
sex = st.number_input("sex", min_value=0, max_value=1, value=0, step=1)
cp = st.number_input("cp", min_value=0, max_value=3, value=0, step=1)
trestbps = st.number_input("trestbps", min_value=94, max_value=200, value=94, step=1)
chol = st.number_input("chol", min_value=127, max_value=600, value=127, step=1)
fbs = st.number_input("fbs", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
restecg = st.number_input("restecg", min_value=0.0, max_value=2.00, value=0.0, step=0.1)
thalach = st.number_input("thalach", min_value=71, max_value=250, value=71, step=1)
exang = st.number_input("exang", min_value=0, max_value=1, value=0, step=1)
oldpeak = st.number_input("oldpeak", min_value=0, max_value=10, value=0, step=1)
slope = st.number_input("slope", min_value=0.0, max_value=2.1, value=0.0, step=0.1)
ca = st.number_input("ca", min_value=0.0, max_value=4.42, value=0.0, step=0.1)
thal = st.number_input("thal", min_value=0, max_value=3, value=0, step=1)

# Create the input dictionary for prediction
input_data = {
  'age':age ,
 'sex': sex,
 'cp': cp,
 'trestbps':trestbps ,
 'chol': chol,
 'fbs': fbs,
 'restecg':restecg ,
 'thalach':thalach ,
  'exang': exang ,
  'oldpeak': oldpeak,
     'slope': slope ,
     'ca': ca,
     'thal': thal
     
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_heart_disease(input_data)

        if pred == 1:
            st.error(f"Prediction: Heart  Disease  with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Heart  Disease   with probability {prob:.2f}")
