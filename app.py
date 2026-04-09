import streamlit as st
import numpy as np
import pickle 



import os

base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, 'model/model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_path, 'model/scaler.pkl'), 'rb'))

st.title("Diabetes Prediction Application")
st.write("Please enter patient disease to predict risk of diabetes")
col1,col2=st.columns(2)
col3,col4=st.columns(2)
col5,col6=st.columns(2)
col7,col8=st.columns(2)



with col1:

    pregnancies=st.number_input("Pregnancies",min_value=0)
with col2:
    Glucose=st.number_input("Glucose",min_value=0)
with col3:
    Bloodpressure=st.number_input("Bloodpressure",min_value=0)
with col4:
    skinthickness=st.number_input("Skinthickness",min_value=0)
with col5:
    Insulin=st.number_input("Insulin",min_value=0)
with col6:
    BMI=st.number_input("BMI",min_value=0)
with col7:
    dpf=st.number_input("DPF",min_value=0)
with col8:
    age=st.number_input("Age",min_value=0)


if st. button("check disease"):
    features=np.array([[pregnancies,Glucose,Bloodpressure,skinthickness,Insulin,BMI,dpf,age]])
    features_scaled=scaler.transform(features)
    predictions=model.predict(features_scaled)[0]
    probability=model.predict_proba(features_scaled)[0][1]*100
    if probability  >= 70:
        risk="High Risk!"
        st.error("High Risk Of Diabetes")
    elif probability >= 40:

        risk="Moderate Risk"
        st.warning("Moderate Risk Of Diabetes")
    else:
        risk='No Risk'
        st.success('No Risk Of Diabetes')
    result='diabetic ' if predictions ==1 else 'Non Diabetic'
    st.subheader ("PREDICTION RESULT")
    st.write(f'RESULT: {result}')

    st.write(f'RISK LEVEL: {risk}')
    st.error(f'probability: {probability:.2f} %')

