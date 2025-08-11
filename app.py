import streamlit as st
import os
import pickle
import docx2txt
import re
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import io

# Load pre-trained model, vectorizer, and label encoder
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

DATA_DIR = "P-561 Dataset"
CSV_PATH = "resume_dataset.csv"

st.set_page_config(page_title="Advanced AI Resume Classifier", layout="wide")
st.title("🤖 Advanced AI Resume Classifier")

# ---------- Helper Functions ----------
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"(\+91[\s-]?)?[6789]\d{9}", text)
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
    tfidf = vectorizer.transform([text])
    pred = model.predict(tfidf)[0]
    return pred, label_encoder.inverse_transform([pred])[0]

def load_all_data():
    data = []
    for category in os.listdir(DATA_DIR):
        cat_path = os.path.join(DATA_DIR, category)
        if os.path.isdir(cat_path):
            for filename in os.listdir(cat_path):
                if filename.endswith(".docx"):
                    text = docx2txt.process(os.path.join(cat_path, filename))
                    details = extract_details(text)
                    details.update({"Filename": filename, "Category": category})
                    data.append(details)
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    return df

# ---------- Section 1: Table Viewer ----------
st.markdown("## 📂 Resume Data Viewer")
category_list = sorted(os.listdir(DATA_DIR))
selected_cat = st.selectbox("Select Category to View Resumes", ["All"] + category_list)

df = load_all_data()
if selected_cat != "All":
    df_filtered = df[df["Category"] == selected_cat]
else:
    df_filtered = df.copy()

st.dataframe(df_filtered)

# ---------- Section 2: Category Visualization ----------
st.markdown("## 📊 Category Distribution")
cat_counts = df["Category"].value_counts()
fig, ax = plt.subplots()
ax.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=140)
st.pyplot(fig)

# ---------- Section 3: Upload Resume for Prediction ----------
st.markdown("---")
st.subheader("📤 Upload Resume for Prediction")

uploaded_file = st.file_uploader("Upload a .docx resume", type=["docx"])
if uploaded_file:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_filepath = tmp_file.name

    text = docx2txt.process(tmp_filepath)

    # Optional: Delete temp file after processing
    try:
        os.remove(tmp_filepath)
    except Exception as e:
        st.warning(f"Could not delete temp file: {e}")

    if not text.strip():
        st.error("❌ Could not extract text from resume.")
    else:
        pred_index, pred_label = predict_category(text)
        details = extract_details(text)
        details["Predicted Category"] = pred_label

        st.success(f"✅ Predicted Category: **{pred_label}**")
        st.write("### 🧾 Extracted Resume Details")
        st.table(pd.DataFrame([details]))

        # Save resume in predicted category folder with unique filename
        save_dir = os.path.join(DATA_DIR, pred_label)
        os.makedirs(save_dir, exist_ok=True)
        unique_filename = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        save_path = os.path.join(save_dir, unique_filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Export prediction to Excel in-memory and provide download button
        result_df = pd.DataFrame([details])
        excel_buffer = io.BytesIO()
        result_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button("📥 Download Prediction as Excel", data=excel_buffer, file_name=f"{uploaded_file.name.split('.')[0]}_result.xlsx")

