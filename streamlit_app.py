import time
import streamlit as st
from openai import OpenAI
from openai.error import RateLimitError

client = OpenAI()

def safe_chat_completion(messages, model="gpt-4o-mini", max_retries=5):
    """Wrapper to handle OpenAI rate limits with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
        except RateLimitError:
            wait_time = min(2 ** attempt, 60)  # exponential backoff up to 60s
            st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    st.error("❌ Failed after multiple retries due to rate limits.")
    return None

# Example Streamlit app usage
st.title("💬 Chatbot with Rate Limit Handling")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = safe_chat_completion(st.session_state.messages)
    
    if response:
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.write(f"**Bot:** {reply}")
