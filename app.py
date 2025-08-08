
import streamlit as st
import os
import pickle
import docx2txt
import re
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained model and objects
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

DATA_DIR = "P-561 Dataset"

st.set_page_config(page_title="AI Resume Classifier", layout="wide")
st.title("🤖 AI Resume Classifier")
st.markdown("Upload or select a resume to classify it into a job category and extract contact details.")

# Extraction functions
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"(\+91[\-\s]?)?[6789]\d{9}", text)
    return match.group(0) if match else "Not found"

def extract_location(text):
    locations = ["Bangalore", "Hyderabad", "Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Visakhapatnam"]
    for loc in locations:
        if loc.lower() in text.lower():
            return loc
    return "Unknown"

def extract_info(field, text):
    match = re.search(rf"{field}[:\s\-]*([A-Za-z0-9 ,&.₹]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "N/A"

def extract_details(text):
    return {
        "Name": extract_info("Name", text),
        "Location": extract_location(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Company": extract_info("Company", text),
        "Skills": extract_info("Skills", text),
        "Experience": extract_info("Experience", text),
        "Salary": extract_info("Salary", text)
    }

def predict_category(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return pred, label_encoder.inverse_transform([pred])[0]

def show_visualization(category_name):
    fig, ax = plt.subplots()
    pd.Series([category_name]).value_counts().plot(kind='bar', ax=ax, color='skyblue')
    plt.title("Predicted Category Distribution")
    st.pyplot(fig)

def display_resume_info(resume_text, category_index):
    pred_label = label_encoder.inverse_transform([category_index])[0]
    st.subheader("📄 Resume Summary")
    st.write(resume_text[:1000] + "...")

    st.subheader("🔍 Prediction")
    st.write(f"**Category:** {pred_label}")

    st.subheader("👤 Candidate Details")
    details = extract_details(resume_text)
    for k, v in details.items():
        icons = {
            "Name": "📛", "Location": "📍", "Email": "📧", "Phone": "📱",
            "Company": "🏢", "Skills": "🛠", "Experience": "📈", "Salary": "💰"
        }
        st.write(f"{icons.get(k, '')} **{k}:** {v}")

    show_visualization(pred_label)

# ========== Dropdown and Random Resume Demo ==========
st.markdown("---")
st.subheader("🎯 Try with a Random Resume")
categories = sorted(os.listdir(DATA_DIR))
selected_category = st.selectbox("Choose a category", categories)

if selected_category:
    resumes = [f for f in os.listdir(os.path.join(DATA_DIR, selected_category)) if f.endswith(".docx")]
    if resumes:
        random_resume = random.choice(resumes)
        resume_path = os.path.join(DATA_DIR, selected_category, random_resume)
        resume_text = docx2txt.process(resume_path)

        st.write(f"**Random Resume from `{selected_category}`**")
        st.download_button("📥 Download Resume", data=open(resume_path, "rb").read(), file_name=random_resume)

        pred_index, pred_label = predict_category(resume_text)
        display_resume_info(resume_text, pred_index)

# ========== Upload Section ==========
st.markdown("---")
st.subheader("📤 Upload Resume")

uploaded_resume = st.file_uploader("Upload a .docx file", type=["docx"])
if uploaded_resume is not None:
    with open(uploaded_resume.name, "wb") as f:
        f.write(uploaded_resume.getbuffer())

    text = docx2txt.process(uploaded_resume.name)
    if text.strip() == "":
        st.error("❌ Couldn't extract text. Try a valid .docx resume.")
    else:
        pred_index, pred_label = predict_category(text)
        display_resume_info(text, pred_index)

        # Save resume to predicted category folder
        save_path = os.path.join(DATA_DIR, pred_label)
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(uploaded_resume.name, os.path.join(save_path, uploaded_resume.name))
        st.success(f"✅ Resume saved to `{pred_label}` folder.")
