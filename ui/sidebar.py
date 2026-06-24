import streamlit as st
from config import FALLBACK_MODELS
from core.loader import load_documents
from core.vectorstore import build_vector_store
from core.chain import build_rag_chain
_FILE_SOURCE = "File (PDF / Word)"
_URL_SOURCE = "Web URL"



def reset_session() -> None:
    """Remove all knowledge sources and reset the session state."""
    for key in ("vector_store", "rag_chain", "messages", "chat_history", "source_label"):
        if key in st.session_state:
            st.session_state.pop(key,None)

def _ingest_source(source_type: str, uploaded_file, url_input: str, api_key: str, model: str) -> None:
    """Load the knowledge source into the vector store and initialize the RAG chain."""

    with st.spinner("Ingesting knowledge source..."):
        try:
            if source_type == _FILE_SOURCE:
                raw_docs = load_documents("file", uploaded_file)
                label = uploaded_file.name
            else:
                raw_docs = load_documents("url", url_input)
                label = url_input.strip()

            vs = build_vector_store(raw_docs, api_key)
            chain = build_rag_chain(vs,api_key,model)

            st.session_state.vector_store = vs
            st.session_state.rag_chain = chain
            st.session_state.source_label = label
            st.session_state.messages =[]
            st.session_state.chat_history =[]

            st.success(f"Successfully ingested knowledge source: {label}", icon="✅")
        
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            
            

def render_sidebar() -> None:
    """Render the sidebar and handle the source ingestion logic"""
    with st.sidebar:
        st.title("Settings")

        api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key here")

        selected_model = st.selectbox("Select Model", FALLBACK_MODELS, index=0,disabled=not api_key)

        st.divider()

        st.subheader("Knowledge Source")

        source_type = st.radio("Input Type", options=[_FILE_SOURCE, _URL_SOURCE], index=0, horizontal=True, label_visibility="collapsed")

        uploaded_file = None
        url_input = None

        if source_type == _FILE_SOURCE:
            uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx","doc"], disabled=not api_key, label_visibility="collapsed")

            can_process = bool(api_key and uploaded_file)
        else:
            url_input = st.text_input("Enter a web URL", placeholder="https://example.com", disabled=not api_key, label_visibility="collapsed")

            can_process = bool(api_key and url_input)
        
        process_button = st.button("Process", disabled=not can_process,type="primary", use_container_width=True)

        if process_button:
            _ingest_source(source_type,uploaded_file,url_input,api_key,selected_model)


        if "source_label" in st.session_state:
            st.divider()
            st.caption(f"Current Source: {st.session_state.source_label}")
            if st.button("Reset Source", type="secondary", use_container_width=True):
                reset_session()
                st.rerun()

        if not api_key:
            st.warning("Please enter your OpenAI API key to enable processing.", icon="⚠️")