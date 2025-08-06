import streamlit as st
import docx2txt
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and other components
with open('resume_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Title and Header
st.title("📄 Resume Classification App")
st.markdown("Classify resumes into one of the predefined job categories using a trained NLP model.")

# Section 1: Choose Category (optional, for display/demo)
st.subheader("Select Expected Category (Optional)")
categories = label_encoder.classes_.tolist()
selected_category = st.selectbox("Choose a category to compare prediction with:", ["-- None --"] + categories)

# Section 2: Resume Upload or Text Input
st.subheader("Upload Resume or Paste Text")
resume_text = ""

upload_option = st.radio("Choose input method:", ("Upload DOCX file", "Paste Resume Text"))

if upload_option == "Upload DOCX file":
    uploaded_file = st.file_uploader("Upload your resume (.docx format)", type=["docx"])
    if uploaded_file:
        resume_text = docx2txt.process(uploaded_file)
        st.success("Resume uploaded successfully!")
        st.text_area("Extracted Resume Text", resume_text, height=200)
else:
    resume_text = st.text_area("Paste your resume text below", "", height=200)

# Section 3: Predict Button
if st.button("🔍 Predict Category"):
    if not resume_text.strip():
        st.warning("Please upload or paste a resume first.")
    else:
        # Transform and predict
        vectorized_input = vectorizer.transform([resume_text])
        prediction = model.predict(vectorized_input)
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        st.markdown("### 🧠 Model Prediction")
        st.success(f"**Predicted Category:** {predicted_category}")

        # Optional: Show if user's selected expected category matches
        if selected_category != "-- None --":
            if selected_category == predicted_category:
                st.info("✅ Matches your selected category!")
            else:
                st.error("❌ Does NOT match your selected category!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit · Resume Classification using NLP")
