import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title("Cutomer Churn Prediction")
#Loading the trained model 
model = load_model('model.h5')
#Loading the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## Loading one hot Encoder
with open('one_hot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

## importing label encoder
with open('label_encoder.pkl','rb') as f:
	label_encoder = pickle.load(f)
     
# Define the Streamlit app
	# Stetting up the streamlit app 
	
geography = st.selectbox('Geography',onehot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('No. of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = {
    'CreditScore': credit_score,
    #'Geography': geography,
    'Gender': label_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_df = pd.DataFrame([input_data])
input_df = pd.concat([input_df, geo_df], axis=1)
input_data_scaled = scaler.transform(input_df)
prediction = model.predict(input_data_scaled)
st.write(f"Prediction Probability: {prediction[0][0]:.4f}")
if prediction[0][0] > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")
