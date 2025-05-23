"""Main entry point for the RAG application."""

import argparse
import os

from utils.helpers import setup_environment_variables
from models.database import db_manager
from services.retriever import chroma_retriever_service
from services.rag_chain import rag_chain_service




def initialize_vector_store():
    """Initialize Chroma vector store with data."""
    # Check if Chroma vector store exists
    from config.settings import VECTOR_STORE_DIR_STR, SCHEMA_CSV_PATH_STR

    # Initialize Chroma if needed
    if not os.path.exists(VECTOR_STORE_DIR_STR):
        print("Initializing Chroma vector store...")
        chroma_retriever_service.load_documents_from_csv(SCHEMA_CSV_PATH_STR)
        print("Chroma vector store initialized.")


def run_interactive_mode():
    """Run the application in interactive mode."""
    print("Using Chroma vector store for retrieval.")

    # Use the RAG chain
    rag_chain = rag_chain_service

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
        "--question",
        type=str,
        help="Question to answer (if not provided, runs in interactive mode)"
    )

    args = parser.parse_args()

    # Initialize vector store
    initialize_vector_store()

    # Print database information
    print(f"Connected to database. Available tables: {db_manager.get_table_names()}")

    # Run in appropriate mode
    if args.question:
        # Single question mode
        # Rebuild the chain with chat prompt
        rag_chain_service._chain = rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
        answer = rag_chain_service.invoke(args.question)

        print("\nAnswer:")
        print(answer)
    else:
        # Interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
