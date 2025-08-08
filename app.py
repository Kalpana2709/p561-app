import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import pickle
import docx2txt
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder

# Load models and encoders
model = pickle.load(open("resume_classifier.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Data directory
DATA_DIR = "P-561 Dataset/Resumes_Docx"
categories = os.listdir(DATA_DIR)

# Extractor functions
def extract_email(text):
    match = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    return match[0] if match else "Not Found"

def extract_phone(text):
    match = re.findall(r"\+?\d[\d\s\-()]{8,}\d", text)
    return match[0] if match else "Not Found"

def extract_location(text):
    # Basic location extractor from pre-defined location keywords
    locations = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"]
    for loc in locations:
        if loc.lower() in text.lower():
            return loc
    return "Not Found"

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.strip().split()) <= 4:
            return line.strip()
    return "Not Found"

def extract_skills(text):
    skills_keywords = ["python", "sql", "java", "machine learning", "data analysis", "excel", "react", "javascript"]
    skills_found = [skill for skill in skills_keywords if skill in text.lower()]
    return ", ".join(skills_found)

def extract_experience(text):
    match = re.findall(r"\d+\+?\s+years", text.lower())
    return match[0] if match else "Not Found"

def extract_salary(text):
    match = re.findall(r"\₹?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:LPA|per annum|pa)?", text.lower())
    return match[0] if match else "Not Found"

# UI Components
st.title("🧠 AI Resume Classifier")
st.markdown("Upload a resume to predict the candidate category and extract useful insights.")

selected_category = st.selectbox("Choose a category to test", options=["Random"] + categories)

uploaded_file = st.file_uploader("Upload a .docx resume", type=["docx"])

if st.button("🔍 Predict Resume") and uploaded_file:
    resume_text = docx2txt.process(uploaded_file)
    
    X_input = tfidf.transform([resume_text])
    pred_class = model.predict(X_input)
    category = le.inverse_transform(pred_class)[0]

    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    location = extract_location(resume_text)
    skills = extract_skills(resume_text)
    exp = extract_experience(resume_text)
    salary = extract_salary(resume_text)

    st.success(f"✅ Predicted Category: {category}")

    st.subheader("📋 Resume Details")
    st.write(f"**Name:** {name}")
    st.write(f"**Email:** {email}")
    st.write(f"**Phone:** {phone}")
    st.write(f"**Location:** {location}")
    st.write(f"**Skills:** {skills if skills else 'Not Found'}")
    st.write(f"**Experience:** {exp}")
    st.write(f"**Expected Salary:** {salary}")

    st.subheader("📝 Resume Summary")
    st.text_area("Summary", resume_text, height=300)

    # Download option
    st.download_button("📥 Download Resume", data=resume_text, file_name=uploaded_file.name)

# Random resume selector for demo
if selected_category != "Random":
    folder_path = os.path.join(DATA_DIR, selected_category)
    all_files = os.listdir(folder_path)
    if st.button("🔁 Show Random Resume from Category"):
        random_file = random.choice(all_files)
        file_path = os.path.join(folder_path, random_file)
        resume_text = docx2txt.process(file_path)

        X_input = tfidf.transform([resume_text])
        pred_class = model.predict(X_input)
        category = le.inverse_transform(pred_class)[0]

        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        location = extract_location(resume_text)
        skills = extract_skills(resume_text)
        exp = extract_experience(resume_text)
        salary = extract_salary(resume_text)

        st.success(f"✅ Predicted Category: {category}")

        st.subheader("📋 Resume Details")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Location:** {location}")
        st.write(f"**Skills:** {skills if skills else 'Not Found'}")
        st.write(f"**Experience:** {exp}")
        st.write(f"**Expected Salary:** {salary}")

        st.subheader("📝 Resume Summary")
        st.text_area("Summary", resume_text, height=300)

# Visualization Placeholder (Optional)
# This can be replaced with actual prediction logs if tracked over time
st.subheader("📊 Category Distribution (Sample)")
category_counts = {cat: len(os.listdir(os.path.join(DATA_DIR, cat))) for cat in categories}
fig, ax = plt.subplots()
ax.bar(category_counts.keys(), category_counts.values())
plt.xticks(rotation=45)
plt.title("Resume Category Distribution")
st.pyplot(fig)
