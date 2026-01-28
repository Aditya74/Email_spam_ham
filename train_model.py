import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
nltk.download('punkt')

# Simple demo dataset (replace later with CSV)
data = {
    "text": [
        "Win a free iPhone now",
        "Hello friend how are you",
        "Congratulations you won a prize",
        "Let's meet tomorrow",
        "Claim your free reward now"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Save BOTH model and vectorizer
with open("spam_email_model.pkl", "wb") as file:
    pickle.dump((model, vectorizer), file)


print("✅ Model trained and saved as spam_email_model.pkl")
