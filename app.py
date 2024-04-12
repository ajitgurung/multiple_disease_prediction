import streamlit as st
import pandas as pd
import joblib

# Load your trained model and any necessary preprocessing steps
model = joblib.load('path_to_your_model.pkl')
# If you have any preprocessing steps (e.g., label encoding), load them here
# encoder = joblib.load('path_to_your_preprocessing_steps.pkl')

# Streamlit App
def main():
    st.title('Disease Prediction App')
    st.sidebar.header('User Input Features')

    # Define a function to collect user inputs
    def user_input_features():
        features = {}
        # Define the input fields based on your model's requirements
        input_fields = [
            'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets',
            'White Blood Cells', 'Red Blood Cells', 'Hematocrit',
            'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin',
            'Mean Corpuscular Hemoglobin Concentration', 'Insulin',
            'BMI', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
            'Triglycerides', 'HbA1c', 'LDL Cholesterol', 'HDL Cholesterol',
            'ALT', 'AST', 'Heart Rate', 'Creatinine', 'Troponin',
            'C-reactive Protein'
        ]
        for field in input_fields:
            features[field] = st.sidebar.slider(field, min_value=0.0, max_value=300.0, value=150.0)
        return pd.DataFrame([features])

    # Get user inputs
    input_df = user_input_features()

    # Display user inputs
    st.write('\n\nUser Input Features:')
    st.write(input_df)

    # Make predictions
    if st.sidebar.button('Predict'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display prediction
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('The patient is predicted to have the disease.')
        else:
            st.write('The patient is predicted to be healthy.')

        # Display prediction probabilities
        st.subheader('Prediction Probability')
        st.write(f'Probability of having the disease: {prediction_proba[0][1]:.2f}')

if __name__ == '__main__':
    main()
