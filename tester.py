import streamlit as st
import machinelearning as ml
import extraction as ex
from bs4 import BeautifulSoup
import requests as re

st.title('Phishing Detection URL test')
choice = st.selectbox("Select the machine learning model",
                      [
                          'AdaBoost',
                          'Decision Tree',
                          'Naive Bayes',
                          'Random Forest',
                          'Support Vector Machine',
                          'Neural Network',
                          'K-Nearest Neighbors'
                      ]
                      )

if choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AB model selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model selected!')
elif choice == 'Naive Bayes':
    model = ml.nb_model
    st.write('NB model selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model selected!')
elif choice == 'Neural Network':
    model = ml.nn_model
    st.write('NN model selected!')
elif choice == 'K-Nearest Neighbors':
    model = ml.kn_model
    st.write('KN model selected!')

url = st.text_input('Enter the chosen URL')
if st.button('Click to Check the website'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print("Could not connect to URL", url)
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [ex.create_vector(soup)]
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This website is legit according to the model's analysis")
            else:
                st.warning('This website is not legit according to the models analysis')
    except re.exceptions.RequestException as e:
        print("= ", e)
