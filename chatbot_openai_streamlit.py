import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4"

# Logging
logging.basicConfig(filename="chat_ui.log", level=logging.INFO)

# Streamlit setup
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

# ======================
# Sidebar controls
# ======================
with st.sidebar:
    st.title("⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", key="dark")
    bg_color = st.color_picker("🎨 Background", "#ffffff")
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()

# ======================
# Global CSS
# ======================
dark_class = "dark-mode" if dark_mode else "light-mode"
st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background-color: {bg_color} !important;
        transition: background-color 0.3s ease;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem 0;
    }}
    .bubble {{
        max-width: 75%;
        padding: 0.8rem 1.2rem;
        border-radius: 1.2rem;
        font-size: 1rem;
        line-height: 1.4;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .user {{
        align-self: flex-end;
        background: #dcf8c6;
        color: #000;
        border-bottom-right-radius: 0;
    }}
    .bot {{
        align-self: flex-start;
        background: #f1f0f0;
        color: #000;
        border-bottom-left-radius: 0;
    }}
    .dark-mode .user {{
        background: #4CAF50;
        color: white;
    }}
    .dark-mode .bot {{
        background: #333;
        color: white;
    }}
    .avatar {{
        font-size: 1.4rem;
        margin-bottom: 0.3rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ======================
# Session state
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sent_time" not in st.session_state:
    st.session_state.last_sent_time = 0

# ======================
# Chat UI
# ======================
st.title("🤖 OpenAI Chatbot")
st.caption("Chat with GPT-4 via Streamlit")

# Render messages
def render_message(role, content):
    role_class = "user" if role == "user" else "bot"
    avatar = "🧑" if role == "user" else "🤖"

    st.markdown(f"""
        <div class="chat-container {dark_class}">
            <div class="bubble {role_class}">
                <div class="avatar">{avatar}</div>
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Show previous messages
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# Typewriter animation
def typewriter(text, delay=0.015):
    output = ""
    placeholder = st.empty()
    for char in text:
        output += char
        placeholder.markdown(f"`{output}`")
        time.sleep(delay)
    placeholder.empty()
    return output

# ======================
# Input & interaction
# ======================
user_input = st.chat_input("Type your message here...")

if user_input:
    now = time.time()
    RATE_LIMIT_SECONDS = 5

    if now - st.session_state.last_sent_time < RATE_LIMIT_SECONDS:
        st.warning("⏱️ Please wait a few seconds.")
    else:
        st.session_state.last_sent_time = now

        # Append user message
        render_message("user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("🤖 Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                reply = response.choices[0].message.content.strip()
                logging.info(f"Bot: {reply}")
        except Exception as e:
            reply = "⚠️ Something went wrong while fetching response."
            logging.exception("OpenAI error")

        animated_reply = typewriter(reply)
        render_message("assistant", animated_reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
