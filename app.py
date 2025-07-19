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
import google.generativeai as genai

# Set page config
st.set_page_config(page_title="AI Mirror", layout="wide")

# Custom CSS to match cosmos.so with responsiveness
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d;
        color: #ffffff;
    }
    .glass-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    @media (max-width: 768px) {
        .glass-box {
            padding: 1rem;
        }
        h1, h2, h3 {
            font-size: 1.5rem !important;
        }
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

# Gemini API integration
api_key = "AIzaSyB0x7jMb8cF0OkKgdnetpzHKIbAgUpkEHQ"
use_gemini = api_key is not None
if use_gemini:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_model()

model_path = "model/personality_model.pkl"
model_loaded = False
if os.path.exists(model_path):
    try:
        personality_model = joblib.load(model_path)
        model_loaded = True
    except:
        st.error("❌ Failed to load the personality model.")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def predict_traits(text):
    try:
        if use_gemini:
            prompt = f"""
            Analyze the following text and estimate the person's Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) on a scale from 0 to 1.

            Text: """{text}"""

            Respond only in this JSON format:
            {{
                "Openness": value,
                "Conscientiousness": value,
                "Extraversion": value,
                "Agreeableness": value,
                "Neuroticism": value
            }}
            """
            response = gemini_model.generate_content(prompt)
            parsed = json.loads(response.text.strip())
            return [parsed["Openness"], parsed["Conscientiousness"], parsed["Extraversion"], parsed["Agreeableness"], parsed["Neuroticism"]]
        else:
            emb = get_bert_embedding(text)
            preds = personality_model.predict(emb)
            return preds[0]
    except:
        st.error("⚠️ Personality estimation failed. Please check your Gemini API key or try again.")
        return [0.5] * 5

def save_feedback(traits, feedback):
    os.makedirs("model", exist_ok=True)
    file_path = "model/user_feedback.json"
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    data.append({"traits": traits, "feedback": feedback, "timestamp": str(datetime.now())})
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

# --- MAIN LAYOUT ---
st.title("🌌 AI Mirror")
st.subheader("Reveal Your Personality Through AI")

with st.container():
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.markdown("### 🔍 What is AI Mirror?")
    st.write("AI Mirror analyzes your words to estimate your Big Five personality traits using cutting-edge AI.")

    st.markdown("### 🧠 How it Works")
    st.write("We use Gemini or BERT to predict your personality. You can either type or speak.")

    st.markdown("### 🚀 Try it Now")
    st.markdown("</div>", unsafe_allow_html=True)

# --- INPUT SECTION ---
with st.container():
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.markdown("## 🧾 Personality Estimator")
    input_option = st.radio("Choose Input Method:", ["Text", "Speech"])

    user_input = ""
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

    if st.button("Predict Traits"):
        if not model_loaded and not use_gemini:
            st.error("🚫 No model or Gemini API key found.")
            st.stop()
        if not user_input.strip():
            st.warning("Please provide some input text or speech.")
        else:
            traits = predict_traits(user_input)
            labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            for i, val in enumerate(traits):
                st.progress(min(1.0, max(0.01, float(val))), text=f"{labels[i]}: {val:.2f}")

            report_path = generate_pdf_report(traits)
            with open(report_path, "rb") as f:
                st.download_button("📄 Download PDF Report", f, file_name="personality_report.pdf")

            st.markdown("#### Was this prediction accurate?")
            feedback = st.radio("Your Feedback:", ["Yes", "No"])
            if st.button("Submit Feedback"):
                save_feedback(traits, feedback)
                st.success("Feedback saved!")
    st.markdown("</div>", unsafe_allow_html=True)

# --- ADMIN ---
with st.expander("🔐 Admin Dashboard"):
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
