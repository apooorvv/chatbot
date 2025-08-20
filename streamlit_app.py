import time
import streamlit as st
from openai import OpenAI
import openai  # needed for error handling
from datetime import datetime

# ✅ Load API key from Streamlit secrets (best for Streamlit Cloud)
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = OpenAI()

# Max number of messages to keep in context
MAX_HISTORY = 6  

def safe_chat_completion(messages, model="gpt-4o-mini", max_retries=5, min_delay=2):
    """Wrapper to handle OpenAI errors with exponential backoff"""
    for attempt in range(max_retries):
        try:
            time.sleep(min_delay)  # ⏱ small delay before each request
            return client.chat.completions.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=500,
            )
        except openai.RateLimitError:
            wait_time = min(2 ** attempt, 60)
            st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.APIError as e:
            wait_time = min(2 ** attempt, 60)
            st.warning(f"⚠️ API error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.APITimeoutError:
            wait_time = min(2 ** attempt, 60)
            st.warning(f"⚠️ API timeout. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.ServiceUnavailableError:
            wait_time = min(2 ** attempt, 60)
            st.warning(f"⚠️ Service unavailable. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    st.error("❌ Failed after multiple retries due to API issues.")
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Apoorv's Chatbot", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 Apoorv's Chatbot</h1>", unsafe_allow_html=True)

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant.", "time": datetime.now().strftime("%I:%M %p")}]

# CSS for chat bubbles + scrollable chat window
st.markdown(
    """
    <style>
    .chat-box {
        max-height: 450px;
        overflow-y: auto;
        padding: 10px;
        border: 2px solid #ddd;
        border-radius: 15px;
        background: #f9f9f9;
    }
    .chat-container {display: flex; align-items: flex-end; margin: 10px 0;}
    .chat-avatar {width: 40px; height: 40px; border-radius: 50%; margin: 5px;}
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        font-size: 16px;
    }
    .timestamp {
        font-size: 12px;
        color: gray;
        margin-top: 3px;
    }
    .user-bubble {
        background-color: #DCF8C6;
        margin-left: auto;
        border-bottom-right-radius: 0;
        text-align: right;
    }
    .bot-bubble {
        background-color: #EAEAEA;
        margin-right: auto;
        border-bottom-left-radius: 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Chat messages inside scrollable box ---
chat_html = "<div class='chat-box'>"
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f"""
        <div class="chat-container" style="justify-content: flex-end;">
            <div>
                <div class="chat-bubble user-bubble">👤 {msg['content']}</div>
                <div class="timestamp">{msg['time']}</div>
            </div>
            <img class="chat-avatar" src="https://cdn-icons-png.flaticon.com/512/847/847969.png">
        </div>
        """
    elif msg["role"] == "assistant":
        chat_html += f"""
        <div class="chat-container" style="justify-content: flex-start;">
            <img class="chat-avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
            <div>
                <div class="chat-bubble bot-bubble">🤖 {msg['content']}</div>
                <div class="timestamp">{msg['time']}</div>
            </div>
        </div>
        """
chat_html += "</div>"

# Render chat
st.markdown(chat_html, unsafe_allow_html=True)

# User input box at bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now().strftime("%I:%M %p")})

    # ✂️ Keep only last MAX_HISTORY messages
    context = st.session_state.messages[-MAX_HISTORY:]

    with st.spinner("Thinking..."):
        response = safe_chat_completion(context)

    if response:
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply, "time": datetime.now().strftime("%I:%M %p")})
        st.experimental_rerun()  # refresh UI

