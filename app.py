import streamlit as st
from PIL import Image
import pickle


# Load the machine learning model
model = pickle.load(open('./Model/Loan_Approval_Prediction.pkl', 'rb'))

# Set page configuration
st.set_page_config(
    page_title="Bank Loan Prediction",
    page_icon="💰",
    layout="wide"
)


# Define the main function to run the web application
def run():
    # Add a title and image to the page
    st.title("Bank Loan Prediction")
    img1 = Image.open('celebal.jpeg')
    img1 = img1.resize((156, 145))
    st.sidebar.image(img1, use_column_width=False)

    # Create a sidebar with project information
    st.sidebar.header("Project Details")
    st.sidebar.markdown("This web application predicts the likelihood of a loan approval by a bank based on the provided information.")

    # Add a form for user inputs
    st.header("User Inputs")
    account_no = st.text_input('Account Number')
    fn = st.text_input('Full Name')

    # Gender
    gen_options = ['Female', 'Male']
    gen = st.selectbox("Gender", gen_options)

    # Marital Status
    mar_options = ['No', 'Yes']
    mar = st.selectbox("Marital Status", mar_options)

    # Dependents
    dep_options = ['No', 'One', 'Two', 'More than Two']
    dep = st.selectbox("Dependents", dep_options)

    # Education
    edu_options = ['Not Graduate', 'Graduate']
    edu = st.selectbox("Education", edu_options)

    # Employment Status
    emp_options = ['Job', 'Business']
    emp = st.selectbox("Employment Status", emp_options)

    # Property Area
    prop_options = ['Rural', 'Semi-Urban', 'Urban']
    prop = st.selectbox("Property Area", prop_options)

    # Credit Score
    cred_options = ['Between 300 to 500', 'Above 500']
    cred = st.selectbox("Credit Score", cred_options)

    # Applicant's Monthly Income
    mon_income = st.number_input("Applicant's Monthly Income ($)", value=0)

    # Co-Applicant's Monthly Income
    co_mon_income = st.number_input("Co-Applicant's Monthly Income ($)", value=0)

    # Loan Amount
    loan_amt = st.number_input("Loan Amount ($)", value=0)

    # Loan Duration
    dur_options = ['2 Months', '6 Months', '8 Months', '1 Year', '16 Months']
    dur = st.selectbox("Loan Duration", dur_options)

    if st.button("Submit"):
        duration = [60, 180, 240, 360, 480]
        dur_mapping = dict(zip(dur_options, duration))
        duration = dur_mapping[dur]

        # Prepare features for prediction
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))

        # Display prediction result
        if ans == 0:
            st.error(f"Hello: {fn} | Account number: {account_no} | According to our analysis, you are unlikely to receive a loan from the bank.")
        else:
            st.success(f"Hello: {fn} | Account number: {account_no} | Congratulations! You are eligible for a loan from the bank.")


# Run the application
run()
