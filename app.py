import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------
# One-time NLTK downloads (safe to run every time; NLTK
# skips re-downloading if already present)
# ---------------------------------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------
# Load model + vectorizer ONCE (cached across reruns)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("tfidf_spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

tfidf_best_model, tfidf = load_model()

# ---------------------------------------------------------
# Same cleaning function used during training
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(tokens)

def predict_email(email):
    cleaned_email = clean_text(email)
    email_vector = tfidf.transform([cleaned_email])
    prediction = tfidf_best_model.predict(email_vector)[0]
    probability = tfidf_best_model.predict_proba(email_vector)[0][1]
    return prediction, probability

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.write(
    "Paste an email below and click **Predict** to check "
    "whether it's spam or not."
)

email_text = st.text_area(
    "Email text",
    height=200,
    placeholder="Paste the email content here..."
)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email text first.")
    else:
        prediction, probability = predict_email(email_text)

        if prediction == 1:
            st.error(f"🚨 Prediction: **SPAM**")
        else:
            st.success(f"✅ Prediction: **NOT SPAM**")

        st.write(f"**Spam Probability:** {round(probability * 100, 2)}%")
        st.progress(float(probability))

st.markdown("---")
st.caption(
    "Model: TF-IDF + Random Oversampling + Multinomial Naive Bayes | Built with Streamlit"
)