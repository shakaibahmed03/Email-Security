import streamlit as st
import string
import base64
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import google.generativeai as genai
import matplotlib.pyplot as plt
import pickle

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer1.pkl', 'rb') as file:
    tfidf = pickle.load(file)

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
    page_title="Spam Detective",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI with dull yellow buttons
st.markdown(
    """
    <style>
    .main {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(255, 0, 0, 0.4);
    }
    .title {
        color: #e60000;
        text-align: center;
        font-weight: bold;
    }
    .stButton button {
        background-color: #d4b000;
        color: white;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(255, 0, 0, 0.4);
    }
    .stButton button:hover {
        background-color: #b09e00;
    }
    .sidebar .sidebar-content {
        background-color: #e60000;
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: white;
    }
    .output.spam {
        text-align: center;
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    .output.not-spam {
        text-align: center;
        color: green;
        font-size: 24px;
        font-weight: bold;
    }
    .title-banner {
        background-color: #e60000;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 32px;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
    }
    .footer {
        background-color: #e60000;
        padding: 10px;
        color: white;
        text-align: center;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.4);
    }
    .login-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .login-title {
        text-align: center;
        color: #e60000;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
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
        page = st.sidebar.radio("Go to", ["Information", "Input", "Output", "History", "Graph"])
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
    st.markdown('<div class="title-banner">Spam Detective</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">Login to Continue</div>
            <p>Enter your credentials to access the Spam Detective application.</p>
            <p>Make sure you have your two-factor authentication device handy.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

# Information page
if page == "Information":
    st.markdown('<div class="title-banner">Spam Detective</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="title-banner">Spam Detective - Input</div>', unsafe_allow_html=True)
    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.session_state['prediction'] = "Spam"
        else:
            st.session_state['prediction'] = "Not Spam"

        st.session_state.history.append((input_sms, st.session_state['prediction'], ""))
        st.write("Prediction complete. Please navigate to the Output page to see the result.")

# Output page
elif page == "Output":
    st.markdown('<div class="title-banner">Spam Detective - Output</div>', unsafe_allow_html=True)
    if 'prediction' in st.session_state:
        prediction_text = st.session_state['prediction']
        if prediction_text == "Spam":
            st.markdown('<div class="output spam">Spam</div>', unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif", width=300)
        else:
            st.markdown('<div class="output not-spam">Not Spam</div>', unsafe_allow_html=True)

        # Generate report using Google Generative AI API
        st.write("Generating report...")

        report = genai.GenerativeModel("gemini-pro").generate_content(
            f"Generate a detailed report on why the following message was classified as {prediction_text}: {st.session_state.history[-1][0]}"
        ).text

        st.write("### Report")
        st.write(report)

        # Add feedback section
        st.write("### Feedback")
        feedback = st.text_input("Please provide your feedback on the prediction:")

        if st.button("Submit Feedback"):
            st.session_state.history[-1] = st.session_state.history[-1][:2] + (feedback,)
            st.success("Thank you for your feedback!")

# History page
elif page == "History":
    st.markdown('<div class="title-banner">Spam Detective - History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.write("Here is the history of your predictions:")
        for i, (text, prediction, feedback) in enumerate(st.session_state.history, start=1):
            st.write(f"**{i}. Message:** {text}")
            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Feedback:** {feedback}")
            st.write("---")
    else:
        st.write("No history available.")

# Graph page
elif page == "Graph":
    st.markdown('<div class="title-banner">Spam Detective - Graph</div>', unsafe_allow_html=True)
    st.write("### Spam vs Non-Spam Distribution")
    
    spam_count = sum(1 for _, pred, _ in st.session_state.history if pred == "Spam")
    not_spam_count = sum(1 for _, pred, _ in st.session_state.history if pred == "Not Spam")

    labels = 'Spam', 'Not Spam'
    sizes = [spam_count, not_spam_count]
    colors = ['#ff6666', '#66b3ff']
    explode = (0.1, 0)  # explode the 1st slice (Spam)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

# Footer
st.markdown(
    """
    <div class="footer">
        A project to classify email as spam or not spam.
    </div>
    """,
    unsafe_allow_html=True
)
