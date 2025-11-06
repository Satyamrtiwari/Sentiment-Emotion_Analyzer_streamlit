import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only first run)
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("logistic_regression_model.pkl")   # ✅ logistic regression model
tfidf = joblib.load("tfidf_vectorizer.pkl")            # ✅ same TF-IDF vectorizer used in training

# Prepare stop words
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(txt):
    txt = txt.lower()  # lowercase
    txt = re.sub(r"http\S+|www\S+", "", txt)  # remove URLs
    txt = txt.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    txt = re.sub(r'\d+', '', txt)  # remove numbers
    words = word_tokenize(txt)  # tokenize
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="centered")

st.title("🧠 Sentiment / Emotion Analysis ")
st.write("Enter text below to analyze its emotion or sentiment.")

# Input box
text = st.text_area("✍️ Enter your text here:")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = preprocess_text(text)
        vector = tfidf.transform([cleaned])   # ✅ works since TF-IDF is fitted
        pred = model.predict(vector)[0]

        # Emotion mapping (based on your earlier dataset)
        emotion_labels = {
            0: 'sadness',
            1: 'anger',
            2: 'love',
            3: 'surprise',
            4: 'fear',
            5: 'joy'
        }

        emotion_name = emotion_labels.get(pred, "Unknown")

        st.success(f"Predicted Emotion: **{emotion_name}**")

        # Optional styling
        if emotion_name in ['joy', 'love']:
            st.markdown("🟢 The sentiment seems **Positive!**")
        elif emotion_name in ['sadness', 'anger', 'fear']:
            st.markdown("🔴 The sentiment seems **Negative!**")
        else:
            st.markdown("🟡 The sentiment seems **Neutral or Mixed.**")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit + ML + NLP")
st.caption("Developed by Satyam Tiwari")

