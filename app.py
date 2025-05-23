"""Streamlit web interface for the RAG application."""

import os
import streamlit as st

from utils.helpers import setup_environment_variables
from models.database import db_manager
from services.retriever import faiss_retriever_service



class RetrieverAdapter:
    """Adapter to make retriever services compatible with RAG chain service."""

    def __init__(self, retriever_service):
        self.retriever_service = retriever_service

    def retrieve(self, query):
        """Retrieve documents and format them for RAG chain."""
        docs = self.retriever_service.retrieve(query)
        # Format the documents as a string for the RAG chain
        formatted_docs = []
        for doc in docs:
            table_name = doc.metadata.get('table', doc.metadata.get('table_name', 'Unknown'))
            formatted_docs.append(f"Table: {table_name}\n{doc.page_content}")
        return "\n\n".join(formatted_docs)


# Set up environment variables
setup_environment_variables()

# Page config
st.set_page_config(page_title="Terrabloom", layout="centered")
st.title("Terrabloom")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for retriever selection
if "retriever_type" not in st.session_state:
    st.session_state.retriever_type = "faiss"

# Initialize session state for conversation state
if "waiting_for_table" not in st.session_state:
    st.session_state.waiting_for_table = False

if "last_ambiguous_question" not in st.session_state:
    st.session_state.last_ambiguous_question = None

# Sidebar for configuration
with st.sidebar:
    st.title("Terrabloom")
    st.header("Configuration")

    # Retriever info
    st.subheader("Retriever Type")
    st.info(f"🚀 Using IVF-FAISS for retrieval")
    st.info(f"📊 {faiss_retriever_service.vector_store.get_retrieval_info()}")

    # Vector Database Management
    st.subheader("Vector Database")

    # Check if FAISS index exists
    faiss_index_exists = os.path.exists("faiss_ivf_index/index.faiss") and os.path.exists("faiss_ivf_index/index.pkl")

    if faiss_index_exists:
        st.success("✅ FAISS index loaded")
        if st.button("🔄 Recreate Index", help="Recreate FAISS index from table_schema.csv"):
            with st.spinner("Recreating FAISS index..."):
                success = faiss_retriever_service.load_documents_from_csv(force_recreate=True)
                if success:
                    st.success("✅ Index recreated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to recreate index")
    else:
        st.warning("⚠️ FAISS index not found")
        if st.button("📥 Load Documents", help="Load documents from table_schema.csv to create FAISS index"):
            with st.spinner("Creating FAISS index..."):
                success = faiss_retriever_service.load_documents_from_csv()
                if success:
                    st.success("✅ Index created successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to create index")

    # Database information
    st.subheader("Database Info")
    st.write(f"Tables: {', '.join(db_manager.get_table_names())}")

    # Conversation state
    st.header("Conversation State")
    if st.session_state.waiting_for_table:
        st.info(f"Waiting for table specification for question: '{st.session_state.last_ambiguous_question}'")

        # Add a button to reset the ambiguous question state
        if st.button("Reset Ambiguous Question"):
            st.session_state.waiting_for_table = False
            st.session_state.last_ambiguous_question = None
            st.rerun()
    else:
        st.success("Ready for new questions")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.waiting_for_table = False
        st.session_state.last_ambiguous_question = None
        st.rerun()

# Display current retriever info
st.info(f"🚀 Currently using: {faiss_retriever_service.vector_store.get_retrieval_info()}")

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

    # Use FAISS retriever and rebuild the chain
    from services.rag_chain import RAGChainService

    st.info("🚀 Using IVF-FAISS retriever for this query")
    faiss_adapter = RetrieverAdapter(faiss_retriever_service)
    rag_chain = RAGChainService(retriever_service=faiss_adapter)
    rag_chain._chain = rag_chain.build_chain(include_schema=True, use_chat_prompt=True)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if we're waiting for a table specification from a previous ambiguous question
                if st.session_state.waiting_for_table and st.session_state.last_ambiguous_question:
                    # Check if the user's response contains a table name
                    tables = db_manager.get_table_names()
                    table_mentioned = next((table for table in tables if table.lower() in prompt.lower()), None)

                    if table_mentioned:
                        # Combine the previous question with the table specification
                        combined_question = f"{st.session_state.last_ambiguous_question} from {table_mentioned}"

                        # Reset the waiting state
                        st.session_state.waiting_for_table = False
                        st.session_state.last_ambiguous_question = None

                        # Process the combined question
                        response = rag_chain.invoke(combined_question)
                    else:
                        # If no table was mentioned, check if the new question is ambiguous
                        is_ambiguous, message = rag_chain.query_service.is_question_ambiguous(prompt)

                        if is_ambiguous:
                            # Keep waiting for a table
                            response = message
                        else:
                            # Not ambiguous, so treat as a new question
                            st.session_state.waiting_for_table = False
                            st.session_state.last_ambiguous_question = None
                            response = rag_chain.invoke(prompt)
                else:
                    # Check if the new question is ambiguous
                    is_ambiguous, message = rag_chain.query_service.is_question_ambiguous(prompt)

                    if is_ambiguous:
                        # Set the waiting state
                        st.session_state.waiting_for_table = True
                        st.session_state.last_ambiguous_question = prompt
                        response = message
                    else:
                        # Not ambiguous, proceed with normal processing
                        response = rag_chain.invoke(prompt)

                # Add the response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.markdown(error_message)
