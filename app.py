import os
import streamlit as st
import json
import requests
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Mirror",
    layout="centered",
    page_icon="🪞"
)

# Load Gemini API key
GEMINI_API_KEY = "AIzaSyB0x7jMb8cF0OkKgdnetpzHKIbAgUpkEHQ"

# Gemini Personality Estimation
def estimate_personality_with_gemini(user_input):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}
    prompt = f"""
You are a psychologist AI mirror.
A user inputs this message: "{user_input}"
Give a concise Big Five personality trait estimation based on it, with scores from 0 to 100.
Respond only in this structured format:

- Openness: XX
- Conscientiousness: XX
- Extraversion: XX
- Agreeableness: XX
- Neuroticism: XX

Also add a 2-line human-style summary.
"""

    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        return text
    except Exception as e:
        return f"⚠️ Failed to get response from Gemini: {str(e)}"

# --- UI Layout ---
st.markdown("""
    <style>
        body {
            background-color: #0f0f0f;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            max-width: 720px;
            margin: auto;
            padding-top: 2rem;
        }
        .stTextArea textarea {
            font-size: 1rem;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🪞 AI Mirror")
st.markdown("#### Reflect your personality through your thoughts")

with st.form("personality_form"):
    user_input = st.text_area("🧠 Enter something about yourself:", placeholder="I enjoy solving problems, meeting new people, and exploring different ideas.", height=150)
    submitted = st.form_submit_button("Analyze")

if submitted and user_input.strip():
    with st.spinner("Analyzing with Gemini..."):
        result = estimate_personality_with_gemini(user_input)
        st.success("Here’s your AI-reflected personality:")
        st.markdown(f"```text\n{result}\n```")

elif submitted:
    st.warning("Please enter something about yourself to analyze.")

st.markdown("---")
st.caption("🔒 Your data stays private | Inspired by [cosmos.so](https://cosmos.so) | Built with ❤️ by AI")
