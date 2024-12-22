import streamlit as st
from PIL import Image
from datetime import datetime
import os
from response import Response
from contextualchathandler import ChatMemory

# Shared instances for global use
chat_handler = ChatMemory()
response_instance = Response(chat_handler)

# Define directories
current_dir = os.path.dirname(__file__)
querylog_path = os.path.join(current_dir, "saved_data", "query_record.txt")

# --- Page Configuration ---
st.set_page_config(
    page_title="Virtual Helper for Intelligent Vehicle Design Module (ENGM298)",
    page_icon="✨",
    layout="wide",
)

# --- Sidebar Styling ---
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: black;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.image("uos-logo.svg", use_container_width=True)

# --- Sidebar Content ---
st.sidebar.markdown(
    """
    ## Tips for Best Results:
    - Be clear and specific with your questions.
    - Mention relevant lecture topics or research papers when possible.
    - Avoid asking vague or overly broad questions.
    """
)

# --- Main Header ---
st.title("💡 The IVDM's Virtual Helper")
st.markdown(
    """
    This tool leverages Retrieval-Augmented Generation (RAG) to answer your questions based on the Intelligent Vehicle Design module content and trusted external resources. Ask your question below!
    """
)

# --- User Input Section ---
user_query = st.text_input("🔍 __Enter your question:__", key="user_query")

# --- Process and Fetch Answer ---
if user_query.strip():
    with st.spinner("Fetching information from sources..."):
        chat_history = "\n".join(chat_handler.get_recent_memory(num_entries=5))
        response_to_query = response_instance.get_RAG_completion(user_query, chat_history)

    # Update memory and log the query
    if response_to_query:
        chat_handler.add_to_memory(user_query, response_to_query)
        st.success("Answer fetched successfully!")
        st.write(response_to_query)

        # Log the query with timestamp
        os.makedirs(os.path.dirname(querylog_path), exist_ok=True)
        with open(querylog_path, "a") as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {user_query}\n")
    else:
        st.error("No relevant information found. Please try rephrasing your query.")

# --- Footer Section ---
st.markdown(
    """
    ---
    *Powered by the University of Surrey. Designed by Prof Saber Fallah to enhance students' learning outcomes and information accessibility.*
    """
)
