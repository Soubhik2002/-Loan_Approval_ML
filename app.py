import streamlit as st
from PIL import Image
import pickle

# Load the machine learning model
model = pickle.load(open('./Model/Loan_Approval_Prediction.pickle', 'rb'))

def preprocess_input():
    # For gender
    gender_options = [0, 1]
    Gender = st.selectbox("Gender", gender_options)

    # For Marital Status
    marital_status_options = [0, 1]
    Married = st.selectbox("Marital Status", marital_status_options)

    # No of dependents
    dependents_options = [0,1,2,3]
    Dependents = st.selectbox("Dependents", dependents_options)

    # For Education
    education_options = [1,0]
    Education = st.selectbox("Education", education_options)

    # For Employment Status
    employment_options = [0,1]
    Self_Employed = st.selectbox("Self Employed ", employment_options)

    # Applicant Monthly Income
    ApplicantIncome = st.number_input("Applicant's Monthly Income($)", value=0)

    # Co-Applicant Monthly Income
    CoapplicantIncome = st.number_input("Co-Applicant's Monthly Income($)", value=0.0)

     # Loan Amount
    LoanAmount = st.number_input("Loan Amount", value=0.0)

    # Loan Duration
    Loan_Amount_Term = st.number_input("Loan Amount Term", value=0)

    # For Credit Score
    credit_score_options = [1.0,0.0]
    Credit_History = st.selectbox("Credit History", credit_score_options)
    
    # For Property Area
    property_area_options = [0,1,2]
    Property_Area = st.selectbox("Property Area", property_area_options)
    
    features = [[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]]
    return features

def main():
    # Load the bank logo image
    img1 = Image.open('celebal.jpeg')
    img1 = img1.resize((156, 145))
    st.image(img1, use_column_width=False)

    st.title("Bank Loan Prediction using Machine Learning")

    features = preprocess_input()
    prediction = model.predict(features)
    if st.button("Submit"):
        if prediction[0] == 0:
            # st.error(
            #     print("According to our Calculations, you will not get the loan from Bank")
            #     )
            print("According to our Calculations, you will not get the loan from Bank")
        else:
            # st.success(
            #         print("Congratulations!! you will get the loan from Bank")
            #     )
            print("Congratulations!! you will get the loan from Bank")

if __name__ == '__main__':
    main()
