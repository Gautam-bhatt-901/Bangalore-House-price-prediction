import numpy as np
import pickle
import pandas as pd
import streamlit as st


clean_df = pd.read_csv('Cleaned_data.csv')
df = pd.read_csv('Bengaluru_House_Data.csv')

model = pickle.load(open("model.pkl", "rb"))

# function for predicting price with trained model
# with these inputs : location, sqft, bath, bhk
def predict(location, sqft, bath, bhk):
    loc_index = np.where(clean_df.columns == location)[0][0]
    x = np.zeros(len(clean_df.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    
    prediction = model.predict([x])[0] * 1e5

    return str(np.round(prediction, 2))

# Title
st.title("Banglore House Price Prediction")

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox(
        'Select a Location',
        options = df['location'].unique()
    )

    bath = st.number_input("Enter total number of Bathrooms:", value = 1, step = 1, min_value = 1)

with col2:
    sqft = st.number_input("Enter Total Square Feet:", value = 300.00, step = 100.0, format = "%.2f", min_value = 300.00)

    bhk = st.number_input("Enter Number of BHK (Bedroom, Hall, Kitchen)", value = 1, step = 1, min_value = 1)

if st.button("Predict"):
    st.write("Prediction: ₹" + predict(location = location, sqft = sqft, bath = bath, bhk = bhk))