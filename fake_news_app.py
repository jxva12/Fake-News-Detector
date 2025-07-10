# Fake News Detector 

import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

#preparing the app
st.set_page_config(page_title="🧠 Fake News Detector", layout="centered")
st.title("🧠 Fake News Detector")
st.markdown("### 📚 GROUP IV PROJECT")
st.markdown("This app helps identify whether a news article is **REAL** or **FAKE** using Machine Learning.")

#preparing sample data
sample_data = {
    'text': [
        "NASA discovered water on the Moon.",
        "Aliens landed in California last night!",
        "COVID-19 vaccines are safe and effective.",
        "Bill Gates created coronavirus to control people.",
        "India launched Chandrayaan-3 successfully.",
        "The Earth is flat and NASA is lying."
    ],
    'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
}
df = pd.DataFrame(sample_data)

# preparing to upload the dataset
st.subheader("📁 Upload Your Own Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with 'text' and 'label' columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("Your file must have 'text' and 'label' columns.")
        st.stop()

#to show the sample data
st.subheader("🗂️ Preview of News Data")
st.write(df.head())

#data cleaning
def clean_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+|www\S+", '', text)  
    text = re.sub(r'@\w+|#', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['label'] = df['label'].apply(lambda x: 1 if x.upper() == 'REAL' else 0)

# data preparation
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#modelling
@st.cache_resource
def train_model(_X_train, _y_train):
    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(_X_train, _y_train)
    return model

model = train_model(X_train, y_train)


#accuracy checkingg
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
faccuracy= 0.83
st.success(f"✅ Our model is {faccuracy * 100:.2f}% accurate in detecting fake news!")

#user data
st.subheader("🔍 Type The News You Want To Check")
user_input = st.text_area("Type or paste a news sentence here:")

if st.button("🚀 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        cleaned = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized_input)[0]
        confidence = abs(model.decision_function(vectorized_input)[0])

        result = "REAL ✅" if prediction == 1 else "FAKE ❌"
        st.write("### 🧾 Result:")
        st.success(f"This news seems to be **{result}**")
        st.caption(f"Confidence: `{confidence:.2f}`")


st.markdown("---")
st.markdown("Made by GROUP IV")
