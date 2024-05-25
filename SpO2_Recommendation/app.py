import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load SVM model
with open('SpO2_Classification.pkl', 'rb') as file:
    model = pickle.load(file)

with open('SpO2_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

with open('SpO2_label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to predict using the SVM model
def predict(features):

    features = np.array(features).reshape(1, -1)

    c_std = sc.transform(features)

    predicted_class = model.predict(c_std)

    return predicted_class



# Streamlit UI
def main():
    st.title('RECOMMENDATION FOR OXYGEN SUPPLEMENTARY')

    # Input features
    feature1 = st.number_input('SpO2 Level')
    feature2 = st.number_input('Age')    

    # Predict button
    if st.button('Predict'):
        features =[feature1, feature2]

        prediction = predict(features)

        predicted_class_decoded = label_encoder.inverse_transform(prediction)

        # Join the predicted class values into a string
        arr_str_flat = ', '.join(predicted_class_decoded.flatten())

        # Print the recommended medicine based on the predicted class
        st.write("Recommended Oxygen Supplementary:", arr_str_flat)


if __name__ == '__main__':
    main()
