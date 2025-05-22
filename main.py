"""Main entry point for the RAG application."""

import argparse
import os

from utils.helpers import setup_environment_variables
from models.database import db_manager
from services.retriever import chroma_retriever_service, qdrant_retriever_service
from services.rag_chain import chroma_rag_chain_service, qdrant_rag_chain_service




def initialize_vector_stores():
    """Initialize vector stores with data."""
    # Check if Chroma vector store exists
    from config.settings import VECTOR_STORE_DIR_STR, SCHEMA_CSV_PATH_STR

    # Initialize Chroma if needed
    if not os.path.exists(VECTOR_STORE_DIR_STR):
        print("Initializing Chroma vector store...")
        chroma_retriever_service.load_documents_from_csv(SCHEMA_CSV_PATH_STR)
        print("Chroma vector store initialized.")

    # Always initialize Qdrant (in-memory)
    print("Initializing Qdrant vector store...")
    qdrant_retriever_service.load_documents_from_csv(SCHEMA_CSV_PATH_STR)
    print("Qdrant vector store initialized.")


def run_interactive_mode(vector_store_type="chroma"):
    """Run the application in interactive mode.

    Args:
        vector_store_type (str, optional): Type of vector store to use.
            Options: "chroma", "qdrant". Defaults to "chroma".
    """
    print(f"Using {vector_store_type} vector store for retrieval.")

    # Select the appropriate RAG chain
    if vector_store_type.lower() == "chroma":
        rag_chain = chroma_rag_chain_service
    elif vector_store_type.lower() == "qdrant":
        rag_chain = qdrant_rag_chain_service
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    print("Interactive mode. Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        print("\nProcessing...")
        try:
            answer = rag_chain.invoke(question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main function."""
    # Set up environment variables
    setup_environment_variables()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG application for SQL database querying")
    parser.add_argument(
        "--vector-store",
        choices=["chroma", "qdrant"],
        default="chroma",
        help="Vector store to use for retrieval (default: chroma)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer (if not provided, runs in interactive mode)"
    )

    args = parser.parse_args()

    # Initialize vector stores
    initialize_vector_stores()

    # Print database information
    print(f"Connected to database. Available tables: {db_manager.get_table_names()}")

    # Run in appropriate mode
    if args.question:
        # Single question mode
        if args.vector_store.lower() == "chroma":
            # Rebuild the chain with chat prompt
            chroma_rag_chain_service._chain = chroma_rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
            answer = chroma_rag_chain_service.invoke(args.question)
        else:
            # Rebuild the chain with chat prompt
            qdrant_rag_chain_service._chain = qdrant_rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
            answer = qdrant_rag_chain_service.invoke(args.question)

        print("\nAnswer:")
        print(answer)
    else:
        # Interactive mode
        run_interactive_mode(args.vector_store)


if __name__ == "__main__":
    main()
