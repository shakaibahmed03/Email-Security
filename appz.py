import streamlit as st
import string
import base64
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import google.generativeai as genai

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')

genai.configure(api_key="AIzaSyDT2XA6oN1XAxYbaSBrE9sD3FQj5ylmrdo")

# Define text preprocessing function
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Email Security through Ensemble Learning",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #4CAF50;
        text-align: center;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 10px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for history and authentication
if 'history' not in st.session_state:
    st.session_state.history = []

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Two-Factor Authentication
if '2fa_enabled' not in st.session_state:
    st.session_state['2fa_enabled'] = False

def login(username, password):
    # Dummy check for example purposes
    if username == "user" and password == "pass":
        st.session_state.logged_in = True
        if st.session_state['2fa_enabled']:
            st.session_state['2fa_verified'] = False
        return True
    else:
        return False

# Sidebar navigation
st.sidebar.title("Navigation")
if st.session_state.logged_in:
    if st.session_state['2fa_enabled'] and not st.session_state.get('2fa_verified', False):
        page = "2FA"
    else:
        page = st.sidebar.radio("Go to", ["Information", "Input", "Output", "History", "Batch Processing", "Admin"])
else:
    page = "Login"

# 2FA page
if page == "2FA":
    st.title("Two-Factor Authentication")
    otp = st.text_input("Enter OTP", type="password")
    totp = pyotp.TOTP("base32secret3232")
    if st.button("Verify"):
        if totp.verify(otp):
            st.session_state['2fa_verified'] = True
            st.experimental_rerun()
        else:
            st.error("Invalid OTP")

# Login page
if page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

# Information page
if page == "Information":
    st.title("Email Security through Ensemble Learning")
    st.write("""
    ## Email/SMS Spam Classifier

    This application allows you to classify emails and SMS messages as spam or not spam.

    **How it works:**
    1. **Preprocessing:** The input text is cleaned and preprocessed.
    2. **Vectorization:** The text is converted into numerical data using TF-IDF vectorization.
    3. **Classification:** The model predicts whether the message is spam or not.

    **Model Details:**
    - The model is trained on a dataset of SMS and email messages.
    - It uses Natural Language Processing (NLP) techniques for preprocessing and feature extraction.

    Navigate to the Input page to classify a message.
    """)

# Input page
elif page == "Input":
    st.title("Input")
    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        prompt = "Classify the following text as either spam or not spam: "
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + input_sms)
        prediction = response.text.strip().lower()
        
        if "not spam" in prediction:
            st.session_state['prediction'] = "Not Spam"
        else:
            st.session_state['prediction'] = "Spam"

        st.session_state.history.append((input_sms, st.session_state['prediction'], ""))
        st.write("Prediction complete. Please navigate to the Output page to see the result.")

# Output page
elif page == "Output":
    st.title("Output")
    if 'prediction' in st.session_state:
        if st.session_state['prediction'] == "Spam":
            st.markdown(
                """
                <style>
                .output {
                    text-align: center;
                    color: red;
                    font-size: 24px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown('<div class="output">Spam</div>', unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif", width=300)
        elif st.session_state['prediction'] == "Not Spam":
            st.markdown(
                """
                <style>
                .output {
                    text-align: center;
                    color: green;
                    font-size: 24px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown('<div class="output">Not Spam</div>', unsafe_allow_html=True)

        # Feedback functionality
        feedback = st.text_input("Enter feedback for the most recent prediction")
        if st.button("Submit Feedback"):
            if feedback:
                st.session_state.history[-1] = (st.session_state.history[-1][0], st.session_state.history[-1][1], feedback)
                st.write("Feedback submitted successfully.")
            else:
                st.error("Please enter feedback.")
    else:
        st.write("No prediction available. Please enter a message on the Input page and click 'Predict'.")

# History page
elif page == "History":
    st.title("History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Message", "Prediction", "Feedback"])
        st.write(df)

        # Option to clear history
        if st.button("Clear History"):
            st.session_state.history.clear()
            st.experimental_rerun()
    else:
        st.write("No history available.")

# Batch Processing page
elif page == "Batch Processing":
    st.title("Batch Processing")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Message' not in df.columns:
            st.error("CSV file must contain a 'Message' column.")
        else:
            predictions = []
            for msg in df['Message']:
                prompt = "Classify the following text as either spam or not spam: "
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(prompt + msg)
                prediction = response.text.strip().lower()
                
                if "not spam" in prediction:
                    predictions.append("Not Spam")
                else:
                    predictions.append("Spam")

            df['Prediction'] = predictions
            st.write(df)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

# Admin page
elif page == "Admin":
    st.title("Admin")
    st.write("Admin functionalities will be implemented here.")
    st.write("Current 2FA status:", "Enabled" if st.session_state['2fa_enabled'] else "Disabled")
    if st.button("Enable 2FA"):
        st.session_state['2fa_enabled'] = True
        st.experimental_rerun()
    elif st.button("Disable 2FA"):
        st.session_state['2fa_enabled'] = False
        st.experimental_rerun()
