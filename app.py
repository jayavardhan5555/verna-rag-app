"""RAG Chat Entry Point for the RAG Chat Application."""

import os

os.environ.setdefault("USER_AGENT","rag-app/1.0")

import streamlit as st
from ui.sidebar import render_sidebar
from ui.chat import render_chat


st.set_page_config(
    page_title="RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

render_sidebar()
render_chat()
