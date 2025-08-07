import streamlit as st
import random
import os
import docx
import pickle
import base64
import matplotlib.pyplot as plt

# Load model and vectorizer
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Resume dataset directory
DATA_DIR = "P-561 Dataset/Resumes_Docx"

st.title("🧠 AI Resume Classifier (Advanced)")
st.write("Upload or select resumes to classify them using AI NLP model")

# Get categories from folders
categories = os.listdir(DATA_DIR)
selected_category = st.selectbox("Choose a Resume Category", categories)

# Get random resume from selected category
resume_files = os.listdir(os.path.join(DATA_DIR, selected_category))
selected_resume = random.choice(resume_files)
resume_path = os.path.join(DATA_DIR, selected_category, selected_resume)

st.subheader(f"📄 Selected Resume: `{selected_resume}`")

# Extract text from .docx
def extract_text(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

resume_text = extract_text(resume_path)
st.text_area("Resume Content", resume_text, height=200)

# Predict
resume_vector = vectorizer.transform([resume_text])
prediction = model.predict(resume_vector)[0]
predicted_label = label_encoder.inverse_transform([prediction])[0]

# Display prediction result
st.success(f"✅ Predicted Category: **{predicted_label}**")

# Generate placeholders (or extract in future)
name = f"{predicted_label.split()[0]} Candidate"
place = "Bangalore"
company = "TechCorp Inc."
skills = ", ".join(predicted_label.split())
experience = f"{random.randint(1, 10)} years"
salary = f"₹{random.randint(3, 15)} LPA"

st.markdown(f"""
### 📌 Resume Metadata
- **Name:** {name}  
- **Place:** {place}  
- **Company:** {company}  
- **Skills:** {skills}  
- **Experience:** {experience}  
- **Previous Salary:** {salary}
""")

# Generate Resume Summary
st.markdown("### 📝 Resume Summary")
st.info(f"This candidate is skilled in {skills}, has {experience} experience, and previously worked at {company} in {place}.")

# Visualization (Dummy pie chart)
st.markdown("### 📊 Category Visualization (Example)")
fig, ax = plt.subplots()
counts = [random.randint(5, 20) for _ in categories]
ax.pie(counts, labels=categories, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# Provide Download link
def create_download_link(path, filename):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Resume</a>'
    return href

st.markdown("### 📎 Download Resume File")
st.markdown(create_download_link(resume_path, selected_resume), unsafe_allow_html=True)
