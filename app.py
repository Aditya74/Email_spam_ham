import streamlit as st
import pickle
import nltk
from preprocessing import clean

# Download nltk data (important for Streamlit Cloud)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load model AND vectorizer
with open("spam_email_model.pkl", "rb") as file:
    model, vectorizer = pickle.load(file)

# App title
st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

# Text input
email_text = st.text_area("Email Text", height=150)

# Predict button
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned_text = clean(email_text)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]

        if prediction == "spam":
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is HAM (Not Spam)")
