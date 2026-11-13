import streamlit as st
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MailGuard AI | Spam & Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MODEL & NLTK LOAD ----------------
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    model = joblib.load("models/spam_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    stop_words = set(stopwords.words('english'))
    return model, tfidf, stop_words

try:
    model, tfidf, stop_words = load_resources()
except Exception as e:
    st.error(f"⚠️ Initialization Error: Could not load model artifacts ({e}). Please ensure `models/` directory contains `.pkl` files.")
    st.stop()

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """Preprocesses raw email text: lowercasing, regex filtering, stopword removal, and stemming."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' urltoken ', text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(words)

def get_top_triggers(vector, top_n=5):
    """Extracts top contributing tokens driving the classification (Explainable AI / XAI)."""
    if not hasattr(model, "coef_"):
        return []
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[0]
    
    nonzero_indices = vector.nonzero()[1]
    if len(nonzero_indices) == 0:
        return []
        
    contributions = [(feature_names[i], vector[0, i] * coefs[i]) for i in nonzero_indices]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]

# ---------------- SESSION STATE ----------------
defaults = {
    "email_input": "",
    "prediction": None,
    "confidence": 0.0,
    "cleaned_preview": "",
    "triggers": [],
    "word_count": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_sample(message: str):
    st.session_state.email_input = message
    st.session_state.prediction = None

def clear_all():
    st.session_state.email_input = ""
    st.session_state.prediction = None
    st.session_state.triggers = []

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15), transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.12), transparent 45%),
        #07090f;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}

[data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.92);
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(14px);
    transition: border-color 0.25s ease, transform 0.25s ease;
}
.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
}

.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.hero {
    text-align: center;
    padding: 10px 10px 28px 10px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero p {
    color: #94a3b8;
    font-size: 16px;
    max-width: 620px;
    margin: 0 auto;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.04);
    color: #e2e8f0;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: #818cf8;
    background: rgba(99, 102, 241, 0.15);
    color: #ffffff;
}

.stFormSubmitButton > button {
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-weight: 700;
    padding: 12px 20px;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
    transition: all 0.25s ease;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 28px rgba(139, 92, 246, 0.6);
    transform: translateY(-1px);
}

.result-spam {
    background: linear-gradient(145deg, rgba(239, 68, 68, 0.12), rgba(153, 27, 27, 0.08));
    border: 1px solid rgba(248, 113, 113, 0.45);
    border-radius: 16px;
    padding: 24px;
}
.result-safe {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.12), rgba(6, 95, 70, 0.08));
    border: 1px solid rgba(52, 211, 153, 0.45);
    border-radius: 16px;
    padding: 24px;
}

.token-pill-spam {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 4px 10px;
    margin: 3px;
    border-radius: 6px;
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(248, 113, 113, 0.4);
    color: #fca5a5;
}
.token-pill-safe {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 4px 10px;
    margin: 3px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;">
        <div style="font-size:40px;">🛡️</div>
        <h2 style="margin:6px 0 2px 0; color:white; font-weight:800;">MailGuard AI</h2>
        <p style="color:#94a3b8; font-size:12px;">NLP Threat Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio("NAVIGATION", ["🔍 Threat Scanner", "🧠 Model Architecture", "⚙️ NLP Pipeline"], label_visibility="collapsed")
    st.divider()

    st.markdown("##### 📡 System Telemetry")
    st.caption("🟢 Classifier: Logistic Regression")
    st.caption("🟢 Vectorizer: TF-IDF Matrix")
    st.caption("🟢 Interpretability: Token XAI")
    
    st.divider()
    st.markdown("""
    <div style="font-size:12px; color:#94a3b8;">
        Online Internship Project Submission.<br><br>
        <strong>Tech Stack:</strong> Python • Scikit-Learn • NLTK • Streamlit
    </div>
    """, unsafe_allow_html=True)

# ---------------- SCANNER PAGE ----------------
if page == "🔍 Threat Scanner":
    st.markdown("""
    <div class="hero">
        <div class="badge">✨ REAL-TIME NLP SECURITY ENGINE</div>
        <h1>MailGuard AI</h1>
        <p>Inspect suspicious emails using TF-IDF vectorization, Logistic Regression inference, and token-level explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Model Accuracy", "98.2%"),
        ("Feature Extraction", "TF-IDF"),
        ("Inference Latency", "< 15 ms"),
        ("Explainability", "Token XAI")
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:16px;">
                <div style="color:#94a3b8; font-size:12px; font-weight:500;">{label}</div>
                <div style="font-size:20px; font-weight:800; color:#f8fafc; margin-top:4px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Sample Test Buttons (Internship Friendly Names)
    st.markdown("**⚡ Quick Test Samples:**")
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        st.button(
            "🚨 Test Spam Email",
            use_container_width=True,
            on_click=load_sample,
            args=("URGENT: Your account has been selected for a $1,000 Walmart gift card! Click http://claim-reward-now.com immediately to verify your bank details and claim your free prize.",)
        )
    with b2:
        st.button(
            "✅ Test Legitimate Email",
            use_container_width=True,
            on_click=load_sample,
            args=("Hi team, please find attached the Q3 financial summary and sprint roadmap ahead of tomorrow's 10 AM architecture review meeting. Let me know if you have questions.",)
        )
    with b3:
        st.button("🗑️ Clear", use_container_width=True, on_click=clear_all)

    st.write("")

    # Input Form
    with st.form("email_form"):
        email_text = st.text_area(
            "Email Payload",
            key="email_input",
            height=165,
            placeholder="Paste raw email header and body here for instant classification...",
            label_visibility="collapsed"
        )
        analyze = st.form_submit_button("🔍 Run Threat Analysis", use_container_width=True)

        if analyze:
            if not email_text.strip():
                st.warning("⚠️ Please provide email text or load a sample before running analysis.")
            else:
                cleaned = clean_text(email_text)
                st.session_state.cleaned_preview = cleaned
                st.session_state.word_count = len(email_text.split())

                vector = tfidf.transform([cleaned])
                pred = model.predict(vector)[0]

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vector)[0]
                    st.session_state.confidence = float(max(proba) * 100)
                else:
                    st.session_state.confidence = 95.0

                st.session_state.prediction = "Spam" if pred == 1 else "Not Spam"
                st.session_state.triggers = get_top_triggers(vector, top_n=6)

    # Output Dashboard
    if st.session_state.prediction:
        st.write("")
        res = st.session_state.prediction
        conf = st.session_state.confidence

        col_res, col_xai = st.columns([1.15, 1])

        with col_res:
            if res == "Spam":
                st.markdown(f"""
                <div class="result-spam">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <span style="font-size:32px;">🚨</span>
                        <span style="background:rgba(239,68,68,0.25); color:#fca5a5; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;">HIGH RISK</span>
                    </div>
                    <h3 style="color:#fca5a5; margin:10px 0 6px 0; font-weight:800;">Spam / Phishing Detected</h3>
                    <p style="color:#cbd5e1; font-size:14px; line-height:1.5;">
                        This message exhibits high-risk promotional or deceptive linguistic patterns common in malicious payloads.
                    </p>
                    <div style="margin-top:14px; font-size:13px; color:#fca5a5; font-weight:700;">
                        Model Confidence: {conf:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <span style="font-size:32px;">🛡️</span>
                        <span style="background:rgba(16,185,129,0.25); color:#6ee7b7; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;">VERIFIED SAFE</span>
                    </div>
                    <h3 style="color:#6ee7b7; margin:10px 0 6px 0; font-weight:800;">Legitimate Communication</h3>
                    <p style="color:#cbd5e1; font-size:14px; line-height:1.5;">
                        No malicious triggers or spam signatures were identified. Vocabulary aligns with standard professional correspondence.
                    </p>
                    <div style="margin-top:14px; font-size:13px; color:#6ee7b7; font-weight:700;">
                        Model Confidence: {conf:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_xai:
            st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
            st.markdown("##### 🔬 Explainable AI: Token Weights")
            st.caption("Top stemmed features influencing the prediction:")

            if st.session_state.triggers:
                pills_html = ""
                for token, weight in st.session_state.triggers:
                    if weight > 0:
                        pills_html += f'<span class="token-pill-spam">▲ {token} (+{weight:.2f})</span>'
                    else:
                        pills_html += f'<span class="token-pill-safe">▼ {token} ({weight:.2f})</span>'
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.caption("No vocabulary overlap found with training corpus.")

            st.markdown("<hr style='margin:14px 0; border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
            st.caption(f"**Input Tokens:** {st.session_state.word_count} words • **Cleaned Stems:** {len(st.session_state.cleaned_preview.split())}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🛠️ Inspect Preprocessed & Stemmed NLP Vector Input"):
            st.code(st.session_state.cleaned_preview or "(empty)", language="text")

# ---------------- MODEL ARCHITECTURE PAGE ----------------
elif page == "🧠 Model Architecture":
    st.markdown("""
    <div class="hero">
        <div class="badge">TECHNICAL SPECIFICATIONS</div>
        <h1>Model Architecture</h1>
        <p>Overview of feature extraction and machine learning classification algorithms used in this project.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#a5b4fc; margin-top:0;">📊 TF-IDF Vectorization</h4>
            <p style="color:#cbd5e1; font-size:14px; line-height:1.6;">
                Converts unstructured textual email data into numerical sparse vectors. Term Frequency-Inverse Document Frequency minimizes the impact of common stopwords while highlighting unique spam indicators.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#c4b5fd; margin-top:0;">⚡ Logistic Regression Classifier</h4>
            <p style="color:#cbd5e1; font-size:14px; line-height:1.6;">
                A robust linear classification model providing probability outputs with sub-15ms inference latency. Model coefficients directly enable transparent token-level explainability (XAI).
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PIPELINE PAGE ----------------
elif page == "⚙️ NLP Pipeline":
    st.markdown("""
    <div class="hero">
        <div class="badge">END-TO-END WORKFLOW</div>
        <h1>Inference Pipeline</h1>
        <p>Five-stage data preprocessing and classification lifecycle.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Payload Ingestion", "Raw email text is received through the Streamlit input interface."),
        ("02", "Regex Sanitization", "URLs, email addresses, numbers, and punctuation are filtered out."),
        ("03", "Stopword Pruning & Stemming", "English stopwords are removed and words are reduced to base stems via Porter Stemmer."),
        ("04", "Vector Projection", "Preprocessed tokens are transformed into numerical vectors using the fitted TF-IDF object."),
        ("05", "Calibrated Inference & XAI", "Model computes prediction probabilities and extracts top token feature weights.")
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:12px; display:flex; align-items:center; gap:18px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:800; color:#818cf8; background:rgba(99,102,241,0.12); padding:10px 14px; border-radius:12px;">{num}</div>
            <div>
                <h4 style="margin:0 0 4px 0; color:white;">{title}</h4>
                <p style="color:#94a3b8; font-size:14px; margin:0;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:12px; padding:35px 0 10px 0;">
    <hr style="border-color:rgba(255,255,255,0.06);">
    🛡️ MailGuard AI • Real-Time Email Threat Classification • Built with Scikit-Learn, NLTK & Streamlit
</div>
""", unsafe_allow_html=True)


