import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Spam Message Classifier")
st.markdown("Developed by [**Saiful Islam Rupom**](https://www.linkedin.com/in/saiful-islam-rupom/)")

input_sms = st.text_area("Enter/Paste the Message/E-mail/SMS below:")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div style='background-color:#ffcccc; padding:8px; border-radius:8px; text-align:center; font-size:30px; font-weight:bold; color:#cc0000;'>Spam</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#ccffcc; padding:8px; border-radius:8px; text-align:center; font-size:30px; font-weight:bold; color:#006600;'>Not Spam</div>",
            unsafe_allow_html=True
        )
