import streamlit as st
import os
import json
import torch
import joblib
import numpy as np
import cv2
from datetime import datetime
from sklearn.linear_model import LinearRegression
from transformers import BertTokenizer, BertModel
from streamlit.components.v1 import html
import speech_recognition as sr
import google.generativeai as genai
from fer import FER

# Set page config
st.set_page_config(page_title="AI Mirror", layout="wide")

# Load models and tokenizer
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

# Configure Gemini
api_key = "AIzaSyB0x7jMb8cF0OkKgdnetpzHKIbAgUpkEHQ"
use_gemini = api_key is not None
if use_gemini:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')

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

def detect_emotion_from_image(uploaded_file):
    detector = FER(mtcnn=True)
    trait_output = [0.5] * 5

    emotion_to_traits = {
        "happy": [0.8, 0.6, 0.9, 0.8, 0.2],
        "sad": [0.5, 0.4, 0.3, 0.6, 0.7],
        "angry": [0.3, 0.5, 0.4, 0.2, 0.8],
        "surprise": [0.9, 0.5, 0.7, 0.5, 0.4],
        "neutral": [0.5, 0.5, 0.5, 0.5, 0.5],
        "disgust": [0.2, 0.6, 0.3, 0.3, 0.7],
        "fear": [0.4, 0.5, 0.4, 0.4, 0.9]
    }

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    result = detector.detect_emotions(frame)
    if result:
        top_emotion, score = detector.top_emotion(frame)
        if top_emotion in emotion_to_traits:
            trait_output = emotion_to_traits[top_emotion]
        st.success(f"Detected emotion: {top_emotion.title()} ({score:.2f})")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)
    else:
        st.info("No face detected. Try another image.")
    return trait_output

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

# --- UI ---
st.title("🧠 AI Mirror - Enhanced")
st.markdown("Estimate your personality using text, speech, or uploaded facial expression image.")

method = st.radio("Choose Input Method", ["Text", "Speech", "Facial Emotion (Image Upload)"])
input_text = ""

if method == "Text":
    input_text = st.text_area("Describe yourself or your experiences")
elif method == "Speech":
    if st.button("🎤 Record Audio"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = recognizer.listen(source)
        try:
            input_text = recognizer.recognize_google(audio)
            st.success(f"You said: {input_text}")
        except:
            st.error("Speech not recognized.")
elif method == "Facial Emotion (Image Upload)":
    uploaded_file = st.file_uploader("Upload an image of your face", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        traits = detect_emotion_from_image(uploaded_file)
        st.markdown("### Personality Traits Estimate:")
        labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        for i, val in enumerate(traits):
            st.progress(val, text=f"{labels[i]}: {val:.2f}")

        feedback = st.radio("Is this accurate?", ["Yes", "No"])
        if st.button("Submit Feedback"):
            save_feedback(traits, feedback)
            st.success("Feedback saved.")

if method in ["Text", "Speech"] and input_text.strip():
    if st.button("🔍 Predict Personality Traits"):
        traits = predict_traits(input_text)
        st.markdown("### Personality Traits Estimate:")
        labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        for i, val in enumerate(traits):
            st.progress(val, text=f"{labels[i]}: {val:.2f}")

        feedback = st.radio("Is this accurate?", ["Yes", "No"])
        if st.button("Submit Feedback"):
            save_feedback(traits, feedback)
            st.success("Feedback saved.")
