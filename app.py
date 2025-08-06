
import streamlit as st
import pickle
import docx2txt
from docx import Document
import fitz  # PyMuPDF
import os

# Load the model, vectorizer, and encoder
with open('resume_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Text extraction function
def extract_text_from_docx(uploaded_file):
    try:
        return docx2txt.process(uploaded_file)
    except Exception as e:
        return f"Error reading DOCX file: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

def predict_resume_category(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    category = le.inverse_transform(prediction)[0]
    return category

# Streamlit UI
st.set_page_config(page_title="Resume Category Classifier", layout="centered")
st.title("🧠 Resume Category Classifier")
st.markdown("Upload a resume file (.docx or .pdf) and get the predicted job category!")

uploaded_file = st.file_uploader("Upload Resume", type=["docx", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if resume_text:
        st.subheader("📄 Extracted Text Preview:")
        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

        if st.button("Predict Category"):
            category = predict_resume_category(resume_text)
            st.success(f"🎯 Predicted Resume Category: **{category}**")
