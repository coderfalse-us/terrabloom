"""Main entry point for the RAG application."""

import argparse
import os

from utils.helpers import setup_environment_variables
from models.database import db_manager
from services.retriever import faiss_retriever_service
from services.rag_chain import rag_chain_service








def run_interactive_mode():
    """Run the application in interactive mode."""
    print("Using FAISS vector store for retrieval.")
    print("⚠️  Make sure to load documents into FAISS index before querying.")

    # Use the default RAG chain service
    rag_chain_service._chain = rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
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



    # Print database information
    print(f"Connected to database. Available tables: {db_manager.get_table_names()}")

    # Run in appropriate mode
    if args.question:
        # Single question mode
        print("⚠️  Make sure to load documents into FAISS index before querying.")
        # Use the default RAG chain service
        rag_chain_service._chain = rag_chain_service.build_chain(include_schema=True, use_chat_prompt=True)
        answer = rag_chain_service.invoke(args.question)

        print("\nAnswer:")
        print(answer)
    else:
        # Interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
