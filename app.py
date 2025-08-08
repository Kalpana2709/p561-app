
import streamlit as st
import os
import pickle
import docx2txt
import re
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Load pre-trained model and vectorizer
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

DATA_DIR = "P-561 Dataset"
OUTPUT_EXCEL = "predicted_resume.xlsx"

st.set_page_config(page_title="AI Resume Classifier", layout="wide")
st.title("🚀 AI Resume Classifier (Advanced Deployment)")

# Extract fields from resume
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
    vec = vectorizer.transform([text])
    cat_index = model.predict(vec)[0]
    return label_encoder.inverse_transform([cat_index])[0]

# Category dropdown
categories = sorted(os.listdir(DATA_DIR))
selected_category = st.selectbox("📂 Select a Category", categories)

# Display resumes in the selected category
st.subheader(f"📄 Resumes in '{selected_category}'")
resumes = [f for f in os.listdir(os.path.join(DATA_DIR, selected_category)) if f.endswith(".docx")]
st.write(f"Total Resumes: {len(resumes)}")
st.table(pd.DataFrame({"File Name": resumes}))

# Pie chart visualization
st.subheader("📊 Category Resume Count")
counts = {cat: len([f for f in os.listdir(os.path.join(DATA_DIR, cat)) if f.endswith('.docx')]) for cat in categories}
fig, ax = plt.subplots()
ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Upload and predict
st.subheader("📤 Upload a Resume for Prediction")
uploaded_resume = st.file_uploader("Upload a .docx file", type=["docx"])

if uploaded_resume:
    with open(uploaded_resume.name, "wb") as f:
        f.write(uploaded_resume.getbuffer())

    text = docx2txt.process(uploaded_resume.name)
    if text.strip() == "":
        st.error("❌ Could not extract any content from the uploaded resume.")
    else:
        predicted_category = predict_category(text)
        details = extract_details(text)
        details["Predicted Category"] = predicted_category

        st.success(f"✅ Resume classified as: {predicted_category}")

        # Save resume to folder
        pred_path = os.path.join(DATA_DIR, predicted_category)
        os.makedirs(pred_path, exist_ok=True)
        shutil.copy(uploaded_resume.name, os.path.join(pred_path, uploaded_resume.name))

        # Show extracted info
        st.write("### Extracted Resume Info")
        st.table(pd.DataFrame([details]))

        # Save to Excel and offer download
        df = pd.DataFrame([details])
        df.to_excel(OUTPUT_EXCEL, index=False)
        with open(OUTPUT_EXCEL, "rb") as xls_file:
            st.download_button("📥 Download Result as Excel", xls_file.read(), file_name=OUTPUT_EXCEL, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
