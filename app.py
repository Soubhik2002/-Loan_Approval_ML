import streamlit as st
import numpy as np  # Importing the NumPy library for numerical operations and array manipulation
import pandas as pd  # Importing the Pandas library for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing the Matplotlib library for creating visualizations and graphs
import sklearn  # Importing scikit-learn, a comprehensive machine learning library
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical variables
from sklearn.linear_model import LogisticRegression  # Importing LogisticRegression for classification
from sklearn.metrics import accuracy_score  # Importing accuracy_score for model evaluation
from sklearn.tree import DecisionTreeClassifier  # Importing DecisionTreeClassifier for decision tree-based classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Importing GradientBoostingClassifier and RandomForestClassifier for ensemble learning
from sklearn.neighbors import KNeighborsClassifier  # Importing KNeighborsClassifier for k-nearest neighbors classification
from sklearn.model_selection import RandomizedSearchCV  # Importing RandomizedSearchCV for hyperparameter tuning
from xgboost import XGBClassifier  # Importing XGBClassifier from the XGBoost library
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier for random forest classification
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.preprocessing import scale, StandardScaler  # Importing scale and StandardScaler for data scaling
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # Importing evaluation metrics
from sklearn.model_selection import cross_val_score  # Importing cross_val_score for cross-validation
import pickle
# Load the trained model
model = pickle.load(open('Loan_Approval_Prediction.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to numeric using LabelEncoder
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])
    data['Dependents'] = le.fit_transform(data['Dependents'])
    
    # Fill missing values with the mean and scale the data
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean())
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mean())
    
    # Scale the data
    scaled_data = pd.DataFrame(scale(data), columns=data.columns)
    return scaled_data

# Streamlit app
def main():
    st.title('Loan Approval Prediction')
    st.write("Enter the details to check if your loan will be approved or not.")
    
    # Input fields for user data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['Yes', 'No'])
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0, value=0)
    coapplicant_income = st.number_input('Co-applicant Income', min_value=0, value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0, value=0)
    credit_history = st.selectbox('Credit History', [0.0, 1.0])
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    
    # Create a DataFrame with the user input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    # Preprocess the input data and make the prediction
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.write("Congratulations! Your loan will be approved.")
    else:
        st.write("Sorry, your loan will not be approved.")
    
if __name__ == '__main__':
    main()
