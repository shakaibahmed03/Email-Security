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
    page_title="Spam Detective",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI with a slightly less red theme
st.markdown(
    """
    <style>
    .main {
        background-color: #fff0e6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(255, 153, 51, 0.4);
    }
    .title {
        color: #cc3300;
        text-align: center;
        font-weight: bold;
    }
    .stButton button {
        background: linear-gradient(90deg, #ff0000, #ff9900, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff);
        color: white;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(255, 153, 51, 0.4);
    }
    .stButton button:hover {
        background-color: #cc3300;
    }
    .sidebar .sidebar-content {
        background-color: #ff9933;
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
        background-color: #ff3300;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 32px;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #666;
        background-color: #f2f2f2;
        border-top: 1px solid #e6e6e6;
        position: fixed;
        width: 100%;
        bottom: 0;
        left: 0;
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

if 'profile' not in st.session_state:
    st.session_state.profile = {"username": "User", "avatar": None}

# Two-Factor Authentication
if '2fa_enabled' not in st.session_state:
    st.session_state['2fa_enabled'] = False

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        .main {
            background-color: #333;
            color: white;
        }
        .stButton button {
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.4);
        }
        .footer {
            background-color: #444;
            color: #ddd;
        }
        </style>
        """, unsafe_allow_html=True
    )

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
st.sidebar.title(f"Welcome, {st.session_state.profile['username']}")
if st.session_state.profile['avatar']:
    st.sidebar.image(st.session_state.profile['avatar'], width=100)

if st.session_state.logged_in:
    if st.session_state['2fa_enabled'] and not st.session_state.get('2fa_verified', False):
        page = "2FA"
    else:
        page = st.sidebar.radio("Go to", ["Information", "Input", "Output", "History", "Profile", "Admin"])
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

    # Save draft option
    if st.button('Save Draft'):
        st.session_state['draft'] = input_sms
        st.write("Draft saved!")

    if 'draft' in st.session_state:
        if st.button('Load Draft'):
            input_sms = st.session_state['draft']
            st.write("Draft loaded!")

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
    st.markdown('<div class="title-banner">Spam Detective - Output</div>', unsafe_allow_html=True)
    if 'prediction' in st.session_state:
        prediction_text = st.session_state['prediction']
        if prediction_text == "Spam":
            st.markdown('<div class="output spam">Spam</div>', unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif", width=300)
        else:
            st.markdown('<div class="output not-spam">Not Spam</div>', unsafe_allow_html=True)

        # Report generation code (unchanged)
        st.write("### Report")
        report = "This is where the report would be generated. Please check the prediction and feedback features."
        st.write(report)

        # Add feedback section
        st.write("### Feedback")
        feedback = st.text_input("Please provide your feedback on the prediction:")
        if st.button("Submit Feedback"):
            st.session_state.history[-1] = (
                st.session_state.history[-1][0],
                st.session_state.history[-1][1],
                feedback,
            )
            st.write("Thank you for your feedback!")
    else:
        st.write("No prediction made yet. Please go to the Input page.")

# History page
elif page == "History":
    st.markdown('<div class="title-banner">Spam Detective - History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        history_df = pd.DataFrame(
            st.session_state.history, columns=["Message", "Prediction", "Feedback"]
        )
        st.write(history_df)
    else:
        st.write("No history available.")

# Profile page
elif page == "Profile":
    st.markdown('<div class="title-banner">Spam Detective - Profile</div>', unsafe_allow_html=True)
    st.image(st.session_state.profile['avatar'] if st.session_state.profile['avatar'] else "https://via.placeholder.com/150", width=150)
    new_username = st.text_input("Username", st.session_state.profile['username'])
    new_avatar = st.file_uploader("Upload Profile Picture")

    if st.button("Update Profile"):
        st.session_state.profile['username'] = new_username
        if new_avatar:
            st.session_state.profile['avatar'] = new_avatar.read()
        st.write("Profile updated successfully!")

# Admin page (simple session timeout warning)
elif page == "Admin":
    st.markdown('<div class="title-banner">Spam Detective - Admin</div>', unsafe_allow_html=True)
    session_timeout = st.slider("Session Timeout (minutes)", 1, 60, 10)
    st.write(f"Session will time out after {session_timeout} minutes of inactivity.")

    if st.button("Enable 2FA"):
        st.session_state['2fa_enabled'] = True
        st.write("Two-Factor Authentication enabled.")

# Footer
st.markdown('<div class="footer">Spam Detective © 2024</div>', unsafe_allow_html=True)
