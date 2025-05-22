"""Streamlit web interface for the RAG application."""

import streamlit as st
import os

from utils.helpers import setup_environment_variables
from models.database import db_manager
from services.retriever import chroma_retriever_service, qdrant_retriever_service
from services.rag_chain import chroma_rag_chain_service, qdrant_rag_chain_service


# Set up environment variables
setup_environment_variables()

# Page config
st.set_page_config(page_title="Terrabloom", layout="centered")
st.title("Terrabloom")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for vector store selection
if "vector_store" not in st.session_state:
    st.session_state.vector_store = "chroma"

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Vector store selection
    vector_store = st.radio(
        "Select Vector Store",
        ["Chroma", "Qdrant"],
        index=0 if st.session_state.vector_store == "chroma" else 1
    )
    st.session_state.vector_store = vector_store.lower()

    # Database information
    st.header("Database Info")
    st.write(f"Tables: {', '.join(db_manager.get_table_names())}")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize vector stores if needed
@st.cache_resource
def initialize_vector_stores():
    """Initialize vector stores with data."""
    from config.settings import VECTOR_STORE_DIR_STR, SCHEMA_CSV_PATH_STR
    import os

    # Initialize Chroma if needed
    if not os.path.exists(VECTOR_STORE_DIR_STR):
        chroma_retriever_service.load_documents_from_csv(SCHEMA_CSV_PATH_STR)

    # Always initialize Qdrant (in-memory)
    qdrant_retriever_service.load_documents_from_csv(SCHEMA_CSV_PATH_STR)

    return True

# Initialize vector stores
initialize_vector_stores()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Select the appropriate RAG chain based on the selected vector store
    if st.session_state.vector_store == "chroma":
        # Rebuild the chain with chat prompt for concise answers
        chroma_rag_chain_service._chain = chroma_rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
        rag_chain = chroma_rag_chain_service
    else:
        # Rebuild the chain with chat prompt for concise answers
        qdrant_rag_chain_service._chain = qdrant_rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
        rag_chain = qdrant_rag_chain_service

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"question": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.markdown(error_message)
