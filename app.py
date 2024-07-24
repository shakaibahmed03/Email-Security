import streamlit as st
import pickle
import string
import base64
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit_authenticator as stauth
import pyotp

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
model = pickle.load(open('model1.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))

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
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.session_state['prediction'] = "Spam"
        else:
            st.session_state['prediction'] = "Not Spam"

        st.session_state.history.append((input_sms, st.session_state['prediction']))
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
        else:
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
    else:
        st.write("No prediction available. Please enter a message on the Input page and click 'Predict'.")

    # Feedback mechanism
    st.write("Please provide your feedback:")
    feedback = st.text_area("Was the prediction correct? Any suggestions?")
    if st.button("Submit Feedback"):
        st.session_state.history[-1] += (feedback,)
        st.success("Thank you for your feedback!")

# History page
elif page == "History":
    st.title("History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Message", "Prediction", "Feedback"])
        st.write(df)

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No history available.")

# Batch Processing page
elif page == "Batch Processing":
    st.title("Batch Processing")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Transformed'] = df['Message'].apply(transform_text)
        df['Prediction'] = model.predict(tfidf.transform(df['Transformed']))
        df['Prediction'] = df['Prediction'].apply(lambda x: "Spam" if x == 1 else "Not Spam")
        st.write(df)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

# Admin page
elif page == "Admin":
    st.title("Admin Dashboard")
    st.write("Here admin users can manage data, retrain models, and more.")
    # Add functionality for uploading new datasets and retraining the model here.
