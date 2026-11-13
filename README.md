# 📧 MailGuard AI — Spam Email Detection

MailGuard AI is a machine learning-based web application that classifies email messages as **Spam** or **Not Spam**. It uses Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression to analyze email content and predict whether a message is suspicious or legitimate.

The application provides a simple and interactive interface built with Streamlit.

## ✨ Features

- Classifies emails as Spam or Not Spam.
- Cleans and preprocesses email text.
- Converts text into numerical features using TF-IDF.
- Uses a trained Logistic Regression model for prediction.
- Provides an interactive Streamlit interface.
- Includes sample emails for quick testing.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Logistic Regression
- Joblib
- Streamlit

## 📊 Model Performance

The Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 98.16% |
| Precision | 95.77% |
| Recall | 98.00% |
| F1-Score | 96.87% |

*Note: These results are based on the test split of the dataset and do not guarantee the same performance on all real-world emails.*

## 📂 Project Structure

```text
spam-email-detection/
│
├── app.py
├── spam_detection.ipynb
├── requirements.txt
│
└── models/
    ├── spam_model.pkl
    └── tfidf_vectorizer.pkl
```

## ⚙️ Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/JiyaSheikh06/spam-email-detection.git
```

2. Navigate to the project directory:

```bash
cd spam-email-detection
```

3. Create and activate a virtual environment (optional):

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Dataset

This project uses the **Spam Mails Dataset** from Kaggle.

Dataset: [Spam Mails Dataset — Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)

## 🧠 Machine Learning Workflow

1. Data Collection and Exploration
2. Text Preprocessing
3. TF-IDF Feature Extraction
4. Model Training
5. Model Evaluation
6. Email Classification

## 🚀 Live Demo

https://spam-detection-app-2026.streamlit.app/

## 👩‍💻 Author

**Javairiya Naeem**

Computer Science Graduate | Aspiring AI Engineer

[GitHub](https://github.com/JiyaSheikh06)

---

If you find this project useful, feel free to ⭐ the repository!