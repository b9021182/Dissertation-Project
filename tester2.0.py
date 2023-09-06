import streamlit as st
import machinelearning as ml
import extraction as ex
from bs4 import BeautifulSoup
import requests as re

st.title('Phishing Detection URL test')

models = {
    'AdaBoost': ml.ab_model,
    'Decision Tree': ml.dt_model,
    'Naive Bayes': ml.nb_model,
    'Random Forest': ml.rf_model,
    'Support Vector Machine': ml.svm_model,
    'Neural Network': ml.nn_model,
    'K-Nearest Neighbors': ml.kn_model
}

url = st.text_input('Enter the chosen URL')

if st.button('Click to Check the website'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            st.error("Could not connect to URL " + url)
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [ex.create_vector(soup)]

            st.subheader('Results:')

            for model_name, model in models.items():
                result = model.predict(vector)
                if result[0] == 0:
                    st.success(f"{model_name}: This website is legit according to the model's analysis")
                else:
                    st.warning(f"{model_name}: This website is not legit according to the model's analysis")

    except re.exceptions.RequestException as e:
        st.error(str(e))
