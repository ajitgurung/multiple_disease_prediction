import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load all your trained models
models = {
    'Gaussian Naive Bayes': joblib.load('models/gnb_model.joblib'),
    'K-Nearest Neighbors': joblib.load('models/knn_model.joblib'),
    'Logistic Regression': joblib.load('models/log_model.joblib'),
    'Support Vector Machine': joblib.load('models/svm_model.joblib')
}

# Define the maximum and minimum values for each feature (scaled)
feature_ranges = {
    'Glucose': (0.0, 1.0),
    'Cholesterol': (0.0, 1.0),
    'Hemoglobin': (0.0, 1.0),
    'Platelets': (0.0, 1.0),
    'White Blood Cells': (0.0, 1.0),
    'Red Blood Cells': (0.0, 1.0),
    'Hematocrit': (0.0, 1.0),
    'Mean Corpuscular Volume': (0.0, 1.0),
    'Mean Corpuscular Hemoglobin': (0.0, 1.0),
    'Mean Corpuscular Hemoglobin Concentration': (0.0, 1.0),
    'Insulin': (0.0, 1.0),
    'BMI': (0.0, 1.0),
    'Systolic Blood Pressure': (0.0, 1.0),
    'Diastolic Blood Pressure': (0.0, 1.0),
    'Triglycerides': (0.0, 1.0),
    'HbA1c': (0.0, 1.0),
    'LDL Cholesterol': (0.0, 1.0),
    'HDL Cholesterol': (0.0, 1.0),
    'ALT': (0.0, 1.0),
    'AST': (0.0, 1.0),
    'Heart Rate': (0.0, 1.0),
    'Creatinine': (0.0, 1.0),
    'Troponin': (0.0, 1.0),
    'C-reactive Protein': (0.0, 1.0)
}

# Streamlit App
def main():
    st.title('Disease Prediction App')
    st.sidebar.header('User Input Features')

    # Define a function to collect user inputs
    def user_input_features():
        features = {}
        for field in feature_ranges:
            min_val, max_val = feature_ranges[field]
            # Set the slider range based on the maximum and minimum values
            features[field] = st.sidebar.slider(field, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2.0)
        return pd.DataFrame([features])

    # Get user inputs
    input_df = user_input_features()

    # Display user inputs
    st.write('\n\nUser Input Features:')
    st.write(input_df)

    # Make predictions with each model when the user clicks the "Predict" button
    if st.sidebar.button('Predict'):
        st.subheader('Disease by: ')

        for model_name, model in models.items():
            prediction = model.predict(input_df)
            print(prediction)
            prediction_proba = None
            
            # Check if the model has predict_proba method for probability estimation
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_df)[0][1]

            # Display prediction result and probability (if available)
            # st.write(f"**{model_name}**: {prediction}", end="")
            if len(prediction) != 0:
                st.write(f"**{model_name}**: {prediction[0]}", end="")
            else:
                st.write('Healthy. :)')

if __name__ == '__main__':
    main()
