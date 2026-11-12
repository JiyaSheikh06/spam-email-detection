
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load trained model and TF-IDF vectorizer
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MailGuard AI | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------

if "email_input" not in st.session_state:
    st.session_state.email_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None


def load_sample(message):
    st.session_state.email_input = message
    st.session_state.prediction = None


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #4338ca55, transparent 35%),
        radial-gradient(circle at 90% 80%, #9333ea44, transparent 35%),
        linear-gradient(135deg, #0f172a, #172554, #312e81);
    color: #f8fafc;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.78);
    border-right: 1px solid rgba(255,255,255,0.12);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}

/* Hero */
.hero {
    text-align: center;
    padding: 30px 15px 35px 15px;
}

.hero-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

/* Labels */
.field-label {
    color: #f1f5f9;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Text area */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.65) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 14px !important;
    font-size: 15px !important;
}

.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    font-weight: 700;
    padding: 12px 20px;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: #c4b5fd;
    box-shadow: 0 0 20px #8b5cf688;
    transform: translateY(-2px);
    color: white;
}

/* Secondary buttons */
div[data-testid="stButton"] button {
    min-height: 42px;
}

/* Metric cards */
.metric-label {
    color: #cbd5e1;
    font-size: 13px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: 750;
    margin-top: 8px;
}

/* Result */
.result-spam {
    background: rgba(239, 68, 68, 0.13);
    border: 1px solid rgba(248, 113, 113, 0.5);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.result-safe {
    background: rgba(16, 185, 129, 0.13);
    border: 1px solid rgba(52, 211, 153, 0.5);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.result-title {
    font-size: 25px;
    font-weight: 800;
    margin: 8px 0;
}

.result-description {
    color: #cbd5e1;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding: 30px 0 10px 0;
}

hr {
    border-color: rgba(255,255,255,0.12);
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0;">
        <div style="font-size:42px;">🛡️</div>
        <h2 style="margin:5px 0;">MailGuard AI</h2>
        <p style="color:#94a3b8; font-size:12px;">
            Intelligent Email Protection
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🧠 About Model", "ℹ️ How It Works"]
    )

    st.divider()

    st.markdown("### 🛡️ Security")
    st.caption("AI-powered email classification")
    st.caption("NLP + Machine Learning")

    st.divider()

    st.caption("Developed as an ML/NLP project")


# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    # Hero section
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🛡️</div>
        <h1>MailGuard AI</h1>
        <p>
            Smart Email Spam Detection System
            <br>
            Protect your inbox with Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)

    metrics = [
        (col1, "🧠", "Detection Engine", "ML + NLP"),
        (col2, "📊", "Text Analysis", "TF-IDF"),
        (col3, "⚡", "Prediction", "Real-time")
    ]

    for col, icon, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size:25px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Email input section
    st.markdown("""
    <div class="glass-card">
        <div class="field-label">✉️ Analyze Your Email</div>
        <p style="color:#cbd5e1; font-size:14px;">
            Paste an email message below to check whether
            it is Spam or Legitimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Sample emails
    st.markdown("**Try a sample email:**")

    sample_col1, sample_col2 = st.columns(2)

    with sample_col1:
        st.button(
            "🚨 Load Spam Example",
            use_container_width=True,
            on_click=load_sample,
            args=("Congratulations! You have won a free prize. "
                  "Click this link now to claim your reward!",)
        )

    with sample_col2:
        st.button(
            "✅ Load Legitimate Example",
            use_container_width=True,
            on_click=load_sample,
            args=("Hello, please find the meeting agenda attached. "
                  "Our meeting is scheduled for tomorrow at 10 AM.",)
        )

    # Input form
    with st.form("email_form"):

        email_text = st.text_area(
            "Email Content",
            key="email_input",
            height=190,
            placeholder="Paste or type your email content here..."
        )

        analyze = st.form_submit_button(
            "🔍 Analyze Email",
            use_container_width=True
        )

    # Prediction using trained ML model
        if analyze:

            if not email_text.strip():
                st.warning("Please enter an email message first.")

            else:
                cleaned_email = clean_text(email_text)

                email_vector = tfidf.transform([cleaned_email])

                prediction = model.predict(email_vector)[0]

                st.session_state.prediction = (
                    "Spam" if prediction == 1 else "Not Spam"
                )

    # Display prediction
    if st.session_state.prediction:

        st.write("")

        result = st.session_state.prediction

        if result == "Spam":

            st.markdown("""
            <div class="result-spam">
                <div style="font-size:40px;">🚨</div>
                <div class="result-title" style="color:#fca5a5;">
                    Spam Email Detected!
                </div>
                <div class="result-description">
                    This message contains suspicious keywords.
                    Treat it with caution.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-safe">
                <div style="font-size:40px;">🛡️</div>
                <div class="result-title" style="color:#6ee7b7;">
                    Legitimate Email
                </div>
                <div class="result-description">
                    No suspicious keywords were detected by
                    this demonstration rule.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Prediction generated using Logistic Regression and TF-IDF.")


# ---------------- ABOUT MODEL PAGE ----------------

elif page == "🧠 About Model":

    st.markdown("""
    <div class="hero">
        <h1>About MailGuard AI</h1>
        <p>Understanding the technology behind email detection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🤖 Machine Learning</h3>
        <p style="color:#cbd5e1;">
            MailGuard AI is designed to classify email messages
            into Spam and Legitimate categories using Natural
            Language Processing and Machine Learning.
        </p>
        <hr>
        <h3>🔤 TF-IDF Vectorization</h3>
        <p style="color:#cbd5e1;">
            TF-IDF converts text into numerical features
            that machine learning algorithms can understand.
        </p>
        <hr>
        <h3>🧠 Classification Algorithm</h3>
        <p style="color:#cbd5e1;">
    Logistic Regression is used as the final classification
    model based on its evaluation performance.
</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HOW IT WORKS PAGE ----------------

elif page == "ℹ️ How It Works":

    st.markdown("""
    <div class="hero">
        <h1>How It Works</h1>
        <p>From raw email text to intelligent prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Email Input", "User enters an email message."),
        ("02", "Text Preprocessing", "Text is cleaned and prepared."),
        ("03", "TF-IDF", "Text is converted into numerical features."),
        ("04", "ML Classification", "The trained model analyzes the features."),
        ("05", "Prediction", "The system displays Spam or Not Spam.")
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:12px;">
            <span style="color:#c4b5fd; font-size:13px;">
                STEP {number}
            </span>
            <h3 style="margin:5px 0;">{title}</h3>
            <p style="color:#cbd5e1; margin-bottom:0;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <hr>
    🛡️ MailGuard AI | Spam Email Detection System
    <br>
    Built with Python, Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
