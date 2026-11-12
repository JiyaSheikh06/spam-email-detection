
import streamlit as st
import joblib
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MailGuard AI | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "models" / "spam_model.pkl")
    tfidf = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
    return model, tfidf

try:
    model, tfidf = load_model()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# ---------------- TEXT PREPROCESSING ----------------

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

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

.stApp {
    background:
        radial-gradient(circle at 5% 5%, #312e8150, transparent 30%),
        radial-gradient(circle at 95% 85%, #0e749050, transparent 30%),
        linear-gradient(135deg, #080f20, #111b35, #171b3b);
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0b1224;
    border-right: 1px solid #263451;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.hero {
    text-align: center;
    padding: 35px 15px 30px;
}

.hero-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
}

.hero p {
    color: #aab8d0;
    font-size: 16px;
    line-height: 1.7;
}

.accent {
    color: #8b9cff;
}

.glass-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 23px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.16);
}

.metric-label {
    color: #aab8d0;
    font-size: 13px;
    margin-top: 8px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: 750;
    margin-top: 5px;
}

.section-label {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.stTextArea textarea {
    background: #111c34 !important;
    color: white !important;
    border: 1px solid #354565 !important;
    border-radius: 13px !important;
    font-size: 15px !important;
}

.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 45px;
    border-radius: 11px;
    border: 1px solid #6475ed;
    background: linear-gradient(100deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 700;
    transition: 0.25s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: #a5b4fc;
    box-shadow: 0 0 20px #6366f155;
    transform: translateY(-2px);
    color: white;
}

.result-card {
    border-radius: 17px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-spam {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(248,113,113,0.45);
}

.result-safe {
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(52,211,153,0.45);
}

.result-title {
    font-size: 26px;
    font-weight: 800;
    margin: 10px 0;
}

.result-description {
    color: #cbd5e1;
    font-size: 14px;
}

.step-number {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #8493ad;
    font-size: 12px;
    padding: 30px 0 10px;
}

hr {
    border-color: rgba(255,255,255,0.12);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:18px 0;">
        <div style="font-size:45px;">🛡️</div>
        <h2 style="margin:5px 0;">MailGuard AI</h2>
        <p style="color:#94a3b8; font-size:12px;">
            Intelligent Email Security
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "🧠 About Model", "⚙️ How It Works"]
    )

    st.divider()

    st.markdown("### Model Information")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Features: TF-IDF")
    st.caption("Task: Binary Classification")

    st.divider()
    st.caption("Built with Python & Streamlit")

# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🛡️</div>
        <h1>MailGuard <span class="accent">AI</span></h1>
        <p>
            Intelligent Email Spam Detection
            <br>
            Analyze email messages with Natural Language Processing
            and Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overview metrics

    col1, col2, col3 = st.columns(3)

    metrics = [
        (col1, "🧠", "Classification Model", "Logistic Regression"),
        (col2, "📊", "Feature Extraction", "TF-IDF"),
        (col3, "⚡", "Detection", "Real-time")
    ]

    for col, icon, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size:26px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:19px;">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Email analyzer

    st.markdown("""
    <div class="glass-card">
        <div class="section-label">EMAIL ANALYZER</div>
        <h3 style="margin-bottom:8px;">Check your email</h3>
        <p style="color:#aab8d0; font-size:14px;">
            Enter an email message below. MailGuard AI will
            analyze its content and classify it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Sample emails

    st.markdown("**Try a sample message**")

    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "🚨 Try Spam Example",
            use_container_width=True,
            on_click=load_sample,
            args=("Congratulations! You have won a free prize. "
                  "Click this link now to claim your reward!",)
        )

    with col2:
        st.button(
            "✅ Try Legitimate Example",
            use_container_width=True,
            on_click=load_sample,
            args=("Hello, please find the meeting agenda attached. "
                  "Our meeting is scheduled for tomorrow at 10 AM.",)
        )

    # Input form

    with st.form("email_form"):

        email_text = st.text_area(
            "Email Message",
            key="email_input",
            height=180,
            placeholder="Paste your email content here..."
        )

        analyze = st.form_submit_button(
            "🔍 Analyze Email",
            use_container_width=True
        )

    # Prediction

    if analyze:

        if not email_text.strip():
            st.session_state.prediction = None
            st.warning("Please enter an email message first.")

        else:
            cleaned_email = clean_text(email_text)
            email_vector = tfidf.transform([cleaned_email])

            prediction = int(model.predict(email_vector)[0])

            # 1 = Spam, 0 = Not Spam
            st.session_state.prediction = (
                "Spam" if prediction == 1 else "Not Spam"
            )

    # Display result

    if st.session_state.prediction:

        result = st.session_state.prediction

        if result == "Spam":

            st.markdown("""
            <div class="result-card result-spam">
                <div style="font-size:42px;">🚨</div>
                <div class="result-title" style="color:#fca5a5;">
                    Spam Email Detected
                </div>
                <div class="result-description">
                    This message has been classified as spam
                    by the trained machine learning model.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-card result-safe">
                <div style="font-size:42px;">🛡️</div>
                <div class="result-title" style="color:#6ee7b7;">
                    Legitimate Email
                </div>
                <div class="result-description">
                    This message has been classified as
                    legitimate by the trained machine learning model.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Powered by Logistic Regression and TF-IDF."
        )

# ---------------- ABOUT MODEL PAGE ----------------

elif page == "🧠 About Model":

    st.markdown("""
    <div class="hero">
        <h1>About <span class="accent">the Model</span></h1>
        <p>Explore the machine learning technology behind MailGuard AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🤖 Logistic Regression</h3>
        <p style="color:#cbd5e1;">
            Logistic Regression is the final classification
            algorithm selected based on its performance during
            model evaluation.
        </p>

        <hr>

        <h3>🔤 TF-IDF Vectorization</h3>
        <p style="color:#cbd5e1;">
            TF-IDF converts cleaned email text into numerical
            features that the machine learning model can process.
        </p>

        <hr>

        <h3>🧹 Text Preprocessing</h3>
        <p style="color:#cbd5e1;">
            Email text is converted to lowercase, URLs and
            special characters are removed, stopwords are filtered,
            and stemming is applied.
        </p>

        <hr>

        <h3>📩 Classification Output</h3>
        <p style="color:#cbd5e1;">
            The model predicts one of two classes:
            Spam or Not Spam (Ham).
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- HOW IT WORKS PAGE ----------------

elif page == "⚙️ How It Works":

    st.markdown("""
    <div class="hero">
        <h1>How It <span class="accent">Works</span></h1>
        <p>A simple journey from email text to machine learning prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Email Input",
         "The user enters an email message for analysis."),

        ("02", "Text Preprocessing",
         "The email text is cleaned and normalized."),

        ("03", "TF-IDF Vectorization",
         "The cleaned text is converted into numerical features."),

        ("04", "Model Classification",
         "The trained Logistic Regression model analyzes the features."),

        ("05", "Final Prediction",
         "The application displays Spam or Not Spam.")
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:14px;">
            <div class="step-number">STEP {number}</div>
            <h3 style="margin:6px 0;">{title}</h3>
            <p style="color:#aab8d0; margin-bottom:0;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <hr>
    🛡️ <b>MailGuard AI</b> | Spam Email Classification
    <br>
    Developed using Python, Streamlit, NLP & Machine Learning
</div>
""", unsafe_allow_html=True)
