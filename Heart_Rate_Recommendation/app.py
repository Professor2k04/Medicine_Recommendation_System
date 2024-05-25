import streamlit as st
import pandas as pd
import numpy as np

import pickle



# Load SVM model
with open('Heart_Beat_Classification.pkl', 'rb') as file:
    model = pickle.load(file)


with open('Heartbeatscaler.pkl', 'rb') as file:
    sc = pickle.load(file)

with open('Heartbeat_label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to predict using the SVM model
def predict(features):

    features = np.array(features).reshape(1, -1)
    c_std = sc.transform(features)
    # Predict the class using the pre-trained SVM model
    predicted_class = model.predict(c_std)

    return predicted_class



# Streamlit UI
def main():
    st.title('MEDICINE RECOMMENDATION FOR BPM')

    # Input features
    feature1 = st.number_input('BPM')
    feature2 = st.number_input('Age')    

    # Predict button
    if st.button('Predict'):
        features =[feature1, feature2]

        prediction = predict(features)

        predicted_class_decoded = label_encoder.inverse_transform(prediction)

        # Join the predicted class values into a string
        arr_str_flat = ', '.join(predicted_class_decoded.flatten())

        # Print the recommended medicine based on the predicted class
        st.write("Recommended Medicine:", arr_str_flat)
        if prediction==0:
            st.write('Recommended Dosage Level: 0.6mg')
        elif prediction==1:
            st.write('The Patient has Ideal Heart Beat Rate')
        else:
            st.write('Recommended Dosage Level: 25mg')
            st.write('Check for Contra Indication-Asthma')
            st.write('If Asthma present' )
            st.write("Recommended Medicine: Atenolol")
            st.write('Recommended Dosage Level: 100mg')




if __name__ == '__main__':
    main()
