import streamlit as st
import pickle
import pandas as pd

# Assuming you have a saved model named 'model.pkl'

# Specify the file path of the saved model
model_path = 'model.pkl'

# Load the model using pickle
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define disorder names
disorder_names = ['Schizophrenia', 'Bipolar disorder', 'Eating disorders',
                  'Anxiety disorders', 'Drug use disorders', 'Depression',
                  'Alcohol use disorders']

# Streamlit app
st.title("Mental Disorder forecaster")

# Country options
country_options = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Canada', 'China',
                   'Denmark', 'England', 'France', 'Germany', 'Greece', 'Hungary',
                   'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Italy', 'Japan',
                   'Malaysia', 'Mexico', 'Netherlands', 'New Zealand',
                   'North America', 'Norway', 'Oceania', 'Pakistan', 'Philippines',
                   'Portugal', 'Qatar', 'Russia', 'Saudi Arabia', 'Singapore',
                   'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sweden',
                   'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'Ukraine',
                   'United Arab Emirates', 'United States', 'Vietnam', 'Zimbabwe']

# User input section
country = st.selectbox("Enter the country:", country_options)
year = st.number_input("Enter the year:", value=2022, min_value=1900, max_value=2100)

# Prediction
if st.button("Predict"):
    # Create a feature input for prediction
    features = [[country, year]]

    # Perform any necessary preprocessing on the features, such as scaling

    # Make predictions using the loaded model
    predictions = model.predict(features)

    # Create a dataframe for the predictions
    predictions_df = pd.DataFrame(predictions, columns=disorder_names)

    # Display the predictions in a table
    st.table(predictions_df)
