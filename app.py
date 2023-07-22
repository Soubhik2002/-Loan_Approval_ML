import streamlit as st
from PIL import Image
import pickle

# Load the machine learning model
model = pickle.load(open('./Model/Loan_Approval_Prediction.pickle', 'rb'))

def preprocess_input():
    # For gender
    gender_options = [0, 1]
    gender = st.selectbox("Gender", gender_options)

    # For Marital Status
    marital_status_options = [0, 1]
    Married = st.selectbox("Marital Status", marital_status_options)

    # No of dependents
    dependents_options = [0,1,2,3]
    Dependents = st.selectbox("Dependents", dependents_options)

    # For Education
    education_options = [1,0]
    education = st.selectbox("Education", education_options)

    # For Employment Status
    employment_options = [0,1]
    Self_Employed = st.selectbox("Employment Status", employment_options)

    # For Property Area
    property_area_options = [0,1,2]
    Property_Area = st.selectbox("Property Area", property_area_options)

    # For Credit Score
    credit_score_options = [1,0]
    Credit_History = st.selectbox("Credit Score", credit_score_options)

    # Applicant Monthly Income
    ApplicantIncome = st.number_input("Applicant's Monthly Income($)", value=0)

    # Co-Applicant Monthly Income
    CoapplicantIncome = st.number_input("Co-Applicant's Monthly Income($)", value=0)

    # Loan Amount
    LoanAmount = st.number_input("Loan Amount", value=0)

    # Loan Duration
     Loan_Amount_Term = st.number_input("Loan Amount Term", value=0)
    # loan_duration_options = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
    # Loan_Amount_Term = st.selectbox("Loan Duration", loan_duration_options)

    # duration = 0
    # if Loan_Amount_Term == '2 Month':
    #     duration = 60
    # elif Loan_Amount_Term == '6 Month':
    #     duration = 180
    # elif Loan_Amount_Term == '8 Month':
    #     duration = 240
    # elif Loan_Amount_Term == '1 Year':
    #     duration = 360
    # elif Loan_Amount_Term == '16 Month':
    #     duration = 480

    features = [[gender,Married,Dependents,education,Self_Employed,Property_Area,Credit_History,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term]]
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

if __name__ == '__main__':
    main()
