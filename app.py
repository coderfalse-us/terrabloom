import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.chain import rag_chain
from config.config import config

# Page configuration
st.set_page_config(
    page_title="TerraBLOOM RAG System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.sub-header {
    font-size: 1.5rem;
    color: #4682B4;
    margin-bottom: 1rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #2E8B57;
    background-color: #f0f8f0;
}

.user-message {
    background-color: #e6f3ff;
    border-left-color: #4682B4;
}

.assistant-message {
    background-color: #f0f8f0;
    border-left-color: #2E8B57;
}

.stats-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🌱 TerraBLOOM RAG System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Intelligent Database Query Assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ System Information</h2>', unsafe_allow_html=True)
    
    # System stats
    try:
        st.subheader("DB Schema Initialization")
        if st.button("Extract Database Schema", help="Extract schemas from all tables in the database"):
            # Show initialization message
            init_message = st.info("🔄Initializing database schema... Please wait.")
            
            try:
                from utils.schema_extract import extract_all_table_schemas
                schemas = extract_all_table_schemas()
                st.session_state.schema_extracted = True
                st.success(f"Successfully extracted {len(schemas) if schemas else 0} table schemas!")
                st.rerun()
            except Exception as e:
                st.error(f"Error extracting schemas: {str(e)}")
            st.success("✅ Done")
            init_message.empty()
        
    # Add a debug section for testing schema retrieval
        if st.checkbox("Debug Mode"):
            st.markdown("### Schema Retrieval Test")
            test_query = st.text_input("Test Query", "Show me information about customers")
            
            if st.button("Test Retriever"):
                from vector_store.retriever import IVFFAISSRetriever
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                
                st.markdown("#### Test Results")
                with st.spinner("Testing retriever..."):
                    try:
                        # Initialize the retriever
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                        retriever = IVFFAISSRetriever()
                        
                        # Get documents
                        documents = retriever.invoke(test_query, k=10)
                        
                        # Display results
                        st.write(f"Retrieved {len(documents)} documents")
                        
                        if documents:
                            # Show first document details
                            st.markdown("#### First Document")
                            st.write(f"Content type: {type(documents[0].page_content)}")
                            st.write(f"Content length: {len(documents[0].page_content)}")
                            st.write(f"Metadata: {documents[0].metadata}")
                            
                            # Show all documents in expandable sections
                            st.markdown("#### All Documents")
                            for i, doc in enumerate(documents):
                                with st.expander(f"Document {i+1} - {doc.metadata.get('type', 'unknown')} - {doc.metadata.get('table', 'unknown')}"):
                                    st.write(doc.page_content)
                                    st.write("Metadata:", doc.metadata)
                        else:
                            st.error("No documents retrieved!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        st.markdown("**Vector Store Statistics:**")
        stats = rag_chain.get_store_stats()
        if "error" not in stats:
            st.metric("Documents", stats.get("document_count", 0))
            st.metric("Embedding Dimension", stats.get("embedding_dim", 0))
            st.metric("Compression Ratio", f"{stats.get('compression_ratio', 0):.2f}x")
        else:
            st.error(stats["error"])
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading stats: {e}")
    
    st.markdown("---")
    
    # Configuration info
    st.markdown("**Configuration:**")
    st.text(f"Host: {config.DB_HOST}:{config.DB_PORT}")
    st.text(f"Database: {config.DB_NAME}")
    st.text(f"User: {config.DB_USER}")
    st.text(f"Schema-Independent: All schemas accessible")
    st.text(f"Model: {config.LLM_MODEL}")
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        rag_chain.clear_history()
        if 'messages' in st.session_state:
            st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = (
        "Welcome to TerraBLOOM RAG System! 🌱\n\n"
        "I can help you query your database using natural language. "
        "Just ask me questions about your data, and I'll generate the appropriate SQL queries and provide answers.\n\n"
        "**Examples:**\n"
        "- How many customers do we have?\n"
        "- Show me all containers with weight capacity over 1000\n"
        "- What are the different types of locations?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.ask_question(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">' 
    "Powered by LangChain, FAISS, and Google Gemini | TerraBLOOM RAG System" 
    "</p>",
    unsafe_allow_html=True
)