import streamlit as st
import pandas as pd
import machinelearning as ml
import extraction as ex
from bs4 import BeautifulSoup
import requests as re

# Load CSV file
urls_df = pd.read_csv("urls.csv")

st.title('Phishing Detection URL test')

# Iterate over the models
models = {
    'AdaBoost': ml.ab_model,
    'Decision Tree': ml.dt_model,
    'Naive Bayes': ml.nb_model,
    'Random Forest': ml.rf_model,
    'Support Vector Machine': ml.svm_model,
    'Neural Network': ml.nn_model,
    'K-Nearest Neighbors': ml.kn_model
}

results = {}

for model_name, model in models.items():
    num_phishing = 0
    num_failed_connections = 0  # New variable to track failed connections

    # Iterate over the URLs in the CSV file
    for url in urls_df['URL']:
        try:
            response = re.get(url, verify=False, timeout=4)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = [ex.create_vector(soup)]
                result = model.predict(vector)
                if result[0] == 1:  # Phishing detection result
                    num_phishing += 1
            else:
                num_failed_connections += 1  # Increment the failed connections counter
        except re.exceptions.RequestException as e:
            num_failed_connections += 1  # Increment the failed connections counter
            continue  # Continue to the next URL in case of error

    results[model_name] = {
        'num_phishing': num_phishing,
        'num_failed_connections': num_failed_connections
    }

# Display the results for each model
for model_name, result in results.items():
    st.write(f"{model_name}: {result['num_phishing']} phishing URLs detected, {result['num_failed_connections']} failed connections")

# Display the total number of URLs
total_urls = len(urls_df)
st.write(f"Total URLs: {total_urls}")