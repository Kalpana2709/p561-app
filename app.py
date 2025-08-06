import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model, vectorizer, and encoder
with open('resume_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Set up Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("📄 AI Resume Classifier")
st.markdown("Upload a resume and choose the expected category to get prediction insights.")

# Dropdown for category selection
category_options = list(encoder.classes_)
selected_category = st.selectbox("🎯 Select Category to Match", category_options)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Resume (.docx format only)", type=['docx'])

if uploaded_file is not None:
    with st.expander("📄 View Uploaded Resume Content"):
        resume_text = docx2txt.process(uploaded_file)
        st.write(resume_text)

    # Predict button
    if st.button("🔍 Predict Category", key="predict_button"):
        input_features = vectorizer.transform([resume_text])
        prediction = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]

        predicted_label = encoder.inverse_transform([prediction])[0]

        st.success(f"✅ **Predicted Category:** `{predicted_label}`")

        # Create a DataFrame for table display (Mock values)
        mock_data = {
            "Name": ["John Doe"],
            "Age": [29],
            "Experience (yrs)": [5],
            "Predicted Category": [predicted_label],
            "Target Category": [selected_category]
        }
        df_result = pd.DataFrame(mock_data)
        st.subheader("📊 Resume Summary")
        st.table(df_result)

        # Bar chart visualization
        st.subheader("📈 Prediction Confidence Scores")
        fig, ax = plt.subplots()
        ax.bar(encoder.classes_, prediction_proba)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Category")
        ax.set_title("Confidence Score per Category")
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("⚠️ Please upload a `.docx` resume to proceed.")
