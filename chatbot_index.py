import json
import requests
import streamlit as st
from streamlit.web import cli as stcli
from icecream import ic

# URLs for your FastAPI endpoints
CHAT_API_URL = "http://127.0.0.1:8000/chat"
RETRIEVE_RECS_API_URL = "http://127.0.0.1:8000/retrieve_recommendations"

# Set up the Streamlit page
st.set_page_config(page_title="Cocktail Advisor Chat", page_icon="🍹")

# Page title
st.title("Cocktail Advisor Chat")

st.markdown(
    """
    Ask any **cocktail-related** question and get recommendations!

    **Note**: This example uses two endpoints:
    - `/retrieve_recommendations` for fetching context from FAISS or user memory
    - `/chat` for generating final LLM-based responses
    """
)

if st.session_state.get("messages", None) is None:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Make username mandatory
username = st.text_input("Your name (required)", key="username")
if not username:
    st.warning("Please enter your name to continue.")
    st.stop()

# Initialize chat history in session state if not existing
if "messages" not in st.session_state:
    st.session_state.messages = []


def retrieve_context_from_api(user_query: str, user_id: str) -> dict:
    """
    Calls '/retrieve_recommendations' to get context for the user query.

    Args:
        user_query (str): User query to retrieve context for (e.g., "I like sweet cocktails").
        user_id (str): Unique user identifier (e.g., "user 123").

    Returns:
        dict: Context data for the user query.
    """
    payload = {"user_id": user_id, "user_query": user_query}
    try:
        resp = requests.post(RETRIEVE_RECS_API_URL, json=payload)
        if resp.status_code == 200:
            data = resp.json()  # Already a dict
            return data["response"]
        else:
            st.error(f"Error retrieving context: {resp.text}")
    except Exception as e:
        st.error(f"Failed to connect to {RETRIEVE_RECS_API_URL}. Error: {str(e)}")

    return {}


def generate_chat_reply(user_query: str, context_obj: dict) -> str:
    """
    Calls '/chat' to get the final chatbot answer.
    We pass 'question', 'context', 'history', and 'model' to the endpoint.
    
    Args:
        user_query (str): User query to generate a response for.
        context_obj (dict): Context data for the user query.
        
    Returns:
        str: Chatbot response to the user query.
    """
    # Convert context to a JSON string for passing as a single text field
    context_str = json.dumps(context_obj)

    # Combine entire chat history into a single string (for optional LLM memory)
    history_str = "\n".join([msg["content"] for msg in st.session_state.messages])

    payload = {
        "question": user_query,
        "context": context_str,
        "history": history_str,
        "model": "gpt-4o",  # or "gpt-4o-mini"
    }

    try:
        resp = requests.post(CHAT_API_URL, json=payload)
        if resp.status_code == 200:
            # The /chat endpoint returns { "response": ... }
            return resp.json().get("response", "No reply received.")
        else:
            st.error(f"Error generating chat reply: {resp.text}")
            return ""
    except Exception as e:
        st.error(f"Failed to connect to {CHAT_API_URL}. Error: {str(e)}")
        return ""


# Create a chat input at the bottom for user queries
user_input = st.chat_input("Enter your question here...")

# If the user entered text, handle it
if user_input:
    # Record the user’s message in the session
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display the user’s message immediately in the chat
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    # 1) Retrieve context from your FAISS + memory backend
    with st.spinner("Retrieving context..."):
        context_data = retrieve_context_from_api(user_input, username)

    # 2) Generate the final LLM response from the /chat endpoint
    with st.spinner("Thinking..."):
        reply_text = generate_chat_reply(user_input, context_data)

    # 3) Add the LLM’s reply to the chat history
    st.session_state.messages.append({"role": "assistant", "content": reply_text})

# Finally, render any existing chat history on page load (before new messages)
st.markdown("### Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.container()
            st.markdown(f"**You:** {msg['content']}")
    else:
        with st.chat_message("assistant"):
            st.container()
            st.markdown(f"**Cocktail Advisor:** {msg['content']}")
