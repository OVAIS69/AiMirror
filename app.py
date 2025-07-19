import streamlit as st
import os
import json
import torch
import joblib
import numpy as np
from fpdf import FPDF
from datetime import datetime
from sklearn.linear_model import LinearRegression
from transformers import BertTokenizer, BertModel
from streamlit.components.v1 import html
import speech_recognition as sr
import hashlib

# Set page config
st.set_page_config(page_title="AI Mirror", layout="wide")

# Inject custom CSS for glassmorphism and cosmic theme
st.markdown("""
    <style>
    body {
        background-color: #0e0e0e;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Particle background
html("""
<div id="particles-js"></div>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 50},
    "color": {"value": "#ffffff"},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.5},
    "size": {"value": 3},
    "line_linked": {"enable": true, "distance": 150, "color": "#ffffff", "opacity": 0.4, "width": 1},
    "move": {"enable": true, "speed": 2}
  }
});
</script>
<style>
#particles-js {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1;
}
</style>
""", height=0)

# Inject animated SVG logo
html("""
<div style="text-align:center; margin-top:-20px;">
<svg width="180" height="60" viewBox="0 0 300 60" fill="none" xmlns="http://www.w3.org/2000/svg">
  <text x="0" y="40" font-size="40" fill="white">
    <tspan fill="url(#grad1)">AI Mirror</tspan>
  </text>
  <defs>
    <linearGradient id="grad1" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#8e2de2" />
      <stop offset="100%" stop-color="#4a00e0" />
    </linearGradient>
  </defs>
</svg>
</div>
""", height=80)

# --- Authentication Setup ---
users_file = "model/users.json"
def load_users():
    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(users_file, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    users = load_users()
    with st.sidebar:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state.user = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password")

        if st.checkbox("New user?"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Register"):
                if new_user in users:
                    st.warning("User already exists")
                else:
                    users[new_user] = hash_password(new_pass)
                    save_users(users)
                    st.success("Registered! Please login.")

login()
if "user" not in st.session_state:
    st.stop()

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_model()

# Load regression model
model_path = "model/personality_model.pkl"
if os.path.exists(model_path):
    personality_model = joblib.load(model_path)
else:
    personality_model = LinearRegression()

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def predict_traits(text):
    emb = get_bert_embedding(text)
    preds = personality_model.predict(emb)
    return preds[0]

def save_feedback(traits, feedback):
    os.makedirs("model", exist_ok=True)
    file_path = "model/user_feedback.json"
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    data.append({"traits": traits.tolist(), "feedback": feedback, "timestamp": str(datetime.now())})
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def generate_pdf_report(traits):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Mirror Personality Report", ln=True, align='C')
    pdf.ln(10)
    for i, trait in enumerate(["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]):
        pdf.cell(200, 10, txt=f"{trait}: {traits[i]:.2f}", ln=True)
    report_path = "model/personality_report.pdf"
    pdf.output(report_path)
    return report_path

# --- LANDING PAGE ---
st.title("🌌 AI Mirror")
st.subheader("Reveal Your Personality Through AI")

with st.container():
    st.markdown("### 🔍 What is AI Mirror?")
    st.write("AI Mirror is an AI-powered tool that analyzes your written or spoken words to estimate your personality traits using advanced language models.")

    st.markdown("### 🧠 How it Works")
    st.write("We use BERT embeddings combined with machine learning regression to predict Big Five traits (OCEAN model). You can type or speak a paragraph and get instant insights.")

    st.markdown("### 🚀 Try it Now")

# --- USER INTERFACE ---
with st.container():
    st.markdown("## 🧾 Personality Estimator")
    input_option = st.radio("Choose Input Method:", ["Text", "Speech"])

    if input_option == "Text":
        user_input = st.text_area("Enter something about yourself:")
    else:
        if st.button("🎤 Record Speech"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Speak now...")
                audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio)
                st.success(f"You said: {user_input}")
            except sr.UnknownValueError:
                st.error("Could not understand audio")
                user_input = ""

    if st.button("Predict Traits") and user_input.strip():
        traits = predict_traits(user_input)
        labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        for i, val in enumerate(traits):
            st.progress(min(1.0, max(0.01, float(val))))
            st.write(f"{labels[i]}: {val:.2f}")

        report_path = generate_pdf_report(traits)
        with open(report_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="personality_report.pdf")

        st.markdown("#### Was this prediction accurate?")
        feedback = st.radio("Your Feedback:", ["Yes", "No"])
        if st.button("Submit Feedback"):
            save_feedback(traits, feedback)
            st.success("Feedback saved!")

# --- ADMIN SECTION ---
with st.expander("🔐 Admin Dashboard"):
    if st.session_state.user == "admin":
        st.markdown("### User Feedback Overview")
        try:
            with open("model/user_feedback.json") as f:
                feedback_data = json.load(f)
            yes_count = sum(1 for d in feedback_data if d['feedback'] == 'Yes')
            no_count = sum(1 for d in feedback_data if d['feedback'] == 'No')
            st.write(f"👍 Yes: {yes_count} | 👎 No: {no_count}")

            if st.button("Retrain Model with Feedback"):
                X = np.array([d['traits'] for d in feedback_data])
                y = X.copy()
                model = LinearRegression()
                model.fit(X, y)
                joblib.dump(model, model_path)
                st.success("Model retrained!")

            st.markdown("### Manage Feedback")
            for i, entry in enumerate(feedback_data):
                st.write(f"{i+1}. Traits: {entry['traits']} | Feedback: {entry['feedback']}")
                if st.button(f"Delete #{i+1}"):
                    feedback_data.pop(i)
                    with open("model/user_feedback.json", "w") as f:
                        json.dump(feedback_data, f, indent=2)
                    st.experimental_rerun()
        except Exception as e:
            st.warning("No feedback found or error loading feedback.")
    else:
        st.info("Only admin can view this section.")

