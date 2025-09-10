# ----------------------------
# app.py - Spam Message Classifier (Streamlit Deploy Ready)
# ----------------------------

import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Step 0: Ensure NLTK works on Streamlit Cloud
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

nltk.data.path.append(nltk_data_dir)
# ----------------------------

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Text transformation function
# ----------------------------
def transform_text(text):
    text = text.lower()  # lowercase
    words = nltk.word_tokenize(text)  # tokenize
    words = [w for w in words if w.isalnum()]  # remove punctuation
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [ps.stem(w) for w in words]  # stemming
    return " ".join(words)

# ----------------------------
# Load model and vectorizer
# ----------------------------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Spam Message Classifier")
st.markdown("Developed by [**Saiful Islam Rupom**](https://www.linkedin.com/in/saiful-islam-rupom/)")

input_sms = st.text_area("Enter/Paste the Message/E-mail/SMS below:")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div style='background-color:#ffcccc; padding:8px; border-radius:8px; "
            "text-align:center; font-size:30px; font-weight:bold; color:#cc0000;'>Spam</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#ccffcc; padding:8px; border-radius:8px; "
            "text-align:center; font-size:30px; font-weight:bold; color:#006600;'>Not Spam</div>",
            unsafe_allow_html=True
        )
