import streamlit as st
from PIL import Image
import pickle

# Load the machine learning model
model = pickle.load(open('./Model/Loan_Approval_Prediction.pickle', 'rb'))

def preprocess_input():
    # For gender
    gender_options = ['Female', 'Male']
    gender = st.selectbox("Gender", gender_options)

    # For Marital Status
    marital_status_options = ['No', 'Yes']
    marital_status = st.selectbox("Marital Status", marital_status_options)

    # No of dependents
    dependents_options = ['No', 'One', 'Two', 'More than Two']
    dependents = st.selectbox("Dependents", dependents_options)

    # For Education
    education_options = ['Not Graduate', 'Graduate']
    education = st.selectbox("Education", education_options)

    # For Employment Status
    employment_options = ['Job', 'Business']
    employment_status = st.selectbox("Employment Status", employment_options)

    # For Property Area
    property_area_options = ['Rural', 'Semi-Urban', 'Urban']
    property_area = st.selectbox("Property Area", property_area_options)

    # For Credit Score
    credit_score_options = ['Between 300 to 500', 'Above 500']
    credit_score = st.selectbox("Credit Score", credit_score_options)

    # Applicant Monthly Income
    applicant_income = st.number_input("Applicant's Monthly Income($)", value=0)

    # Co-Applicant Monthly Income
    coapplicant_income = st.number_input("Co-Applicant's Monthly Income($)", value=0)

    # Loan Amount
    loan_amount = st.number_input("Loan Amount", value=0)

    # Loan Duration
    loan_duration_options = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
    loan_duration = st.selectbox("Loan Duration", loan_duration_options)

    duration = 0
    if loan_duration == '2 Month':
        duration = 60
    elif loan_duration == '6 Month':
        duration = 180
    elif loan_duration == '8 Month':
        duration = 240
    elif loan_duration == '1 Year':
        duration = 360
    elif loan_duration == '16 Month':
        duration = 480

    features = [[gender, marital_status, dependents, education, employment_status,
                 applicant_income, coapplicant_income, loan_amount, duration,
                 credit_score, property_area]]
    return features

def main():
    # Load the bank logo image
    img1 = Image.open('celebal.jpeg')
    img1 = img1.resize((156, 145))
    st.image(img1, use_column_width=False)

    st.title("Bank Loan Prediction using Machine Learning")

    ## Account No
    account_no = st.text_input('Account number')

    ## Full Name
    fn = st.text_input('Full Name')

    if st.button("Submit"):
        features = preprocess_input()
        prediction = model.predict(features)
        if prediction[0] == 0:
            st.error(
                f"Hello: {fn} || Account number: {account_no} || According to our Calculations, you will not get the loan from Bank"
            )
        else:
            st.success(
                f"Hello: {fn} || Account number: {account_no} || Congratulations!! you will get the loan from Bank"
            )

if _name_ == '_main_':
    main()
