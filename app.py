import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load your trained model and any necessary preprocessing steps
model = joblib.load('models/svm_model.joblib')


# If you have any preprocessing steps (e.g., StandardScaler), load them here
# scaler = joblib.load('path_to_your_preprocessing_steps.pkl')

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
        # Preprocess input data if necessary (e.g., scaling)
        # input_df_scaled = scaler.transform(input_df)  # Apply preprocessing steps

        # Use the model to make predictions
        prediction = model.predict(input_df)

        # Display prediction
        st.subheader('Prediction')
        if len(prediction) != 0:
            st.write(f'The patient is suffering from: {prediction}.')
        else:
            st.write('The patient is predicted to be healthy.')

if __name__ == '__main__':
    main()
