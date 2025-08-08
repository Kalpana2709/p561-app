# app.py
import os
import random
import re
import docx2txt
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

DATA_DIR = "P-561 Dataset"

def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

def extract_details(text):
    name_match = re.findall(r"Name[:\s]+([A-Za-z .]+)", text)
    email_match = re.findall(r"[\w.-]+@[\w.-]+", text)
    phone_match = re.findall(r"\+?\d[\d -]{8,}\d", text)
    location_match = re.findall(r"Location[:\s]+([A-Za-z ,]+)", text)
    company_match = re.findall(r"Company[:\s]+([A-Za-z0-9 &.]+)", text)
    skills_match = re.findall(r"Skills[:\s]+([A-Za-z0-9 ,]+)", text)
    experience_match = re.findall(r"Experience[:\s]+([0-9]+\s+years)", text)
    salary_match = re.findall(r"Salary[:\s]+([A-Za-z0-9 ₹.,]+)", text)

    return {
        "Name": name_match[0] if name_match else "N/A",
        "Email": email_match[0] if email_match else "N/A",
        "Phone": phone_match[0] if phone_match else "N/A",
        "Location": location_match[0] if location_match else "N/A",
        "Company": company_match[0] if company_match else "N/A",
        "Skills": skills_match[0] if skills_match else "N/A",
        "Experience": experience_match[0] if experience_match else "N/A",
        "Salary": salary_match[0] if salary_match else "N/A",
    }

def predict_category(text):
    tfidf = vectorizer.transform([text])
    return model.predict(tfidf)[0]

def show_resume_details(text):
    details = extract_details(text)
    st.write("### Candidate Details")
    for key, val in details.items():
        st.write(f"**{key}:** {val}")
    return details

# Streamlit UI
st.set_page_config(page_title="AI Resume Classifier", layout="wide")
st.title("📄 AI Resume Classifier")

categories = sorted(os.listdir(DATA_DIR))
category = st.selectbox("Select a Category", categories)

if category:
    category_path = os.path.join(DATA_DIR, category)
    resumes = [f for f in os.listdir(category_path) if f.endswith(".docx")]
    if resumes:
        random_resume = random.choice(resumes)
        resume_path = os.path.join(category_path, random_resume)
        resume_text = extract_text_from_docx(resume_path)

        st.subheader(f"Random Resume from '{category}': {random_resume}")
        st.download_button("📥 Download Resume", data=open(resume_path, "rb").read(), file_name=random_resume)

        st.write("### Resume Summary")
        st.write(resume_text[:1000] + "...")

        predicted_category = predict_category(resume_text)
        st.write(f"### 🔍 Predicted Category: **{predicted_category}**")

        details = show_resume_details(resume_text)

        # Category Distribution Visualization
        category_counts = pd.Series([predicted_category])
        fig, ax = plt.subplots()
        category_counts.value_counts().plot(kind='bar', ax=ax)
        plt.title("Predicted Category Distribution")
        st.pyplot(fig)

uploaded_file = st.file_uploader("Upload a Resume (.docx)", type=["docx"])
if uploaded_file is not None:
    resume_text = extract_text_from_docx(uploaded_file)
    st.write("### Uploaded Resume Summary")
    st.write(resume_text[:1000] + "...")

    predicted_category = predict_category(resume_text)
    st.write(f"### 🔍 Predicted Category: **{predicted_category}**")

    details = show_resume_details(resume_text)

    # Save uploaded resume
    save_path = os.path.join(DATA_DIR, predicted_category)
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, uploaded_file.name)
    with open(save_file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Saved uploaded resume to {save_path}")
