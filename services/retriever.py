"""Vector store and retrieval functionality for the RAG application."""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import (
    VECTOR_STORE_DIR,
    COLLECTION_NAME,
    SCHEMA_CSV_PATH,
    RETRIEVER_K
)
from models.embeddings import embedding_manager


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create_chroma(persist_directory=None, collection_name=None, embedding_function=None):
        """Create a Chroma vector store.

        Args:
            persist_directory (str, optional): Directory to persist the vector store.
                Defaults to VECTOR_STORE_DIR from settings.
            collection_name (str, optional): Name of the collection.
                Defaults to COLLECTION_NAME from settings.
            embedding_function (object, optional): Embedding function to use.
                Defaults to embedding_manager.embeddings.

        Returns:
            Chroma: The Chroma vector store.
        """
        persist_directory = persist_directory or VECTOR_STORE_DIR
        collection_name = collection_name or COLLECTION_NAME
        embedding_function = embedding_function or embedding_manager.embeddings

        # Convert Path object to string to avoid TypeError
        if hasattr(persist_directory, 'as_posix'):
            persist_directory = str(persist_directory)

        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )




class RetrieverService:
    """Service for document retrieval."""

    def __init__(self, k=None):
        """Initialize the retriever service.

        Args:
            k (int, optional): Number of documents to retrieve.
                Defaults to RETRIEVER_K from settings.
        """
        self.k = k or RETRIEVER_K
        self._vector_store = None
        self._retriever = None

    def _create_vector_store(self):
        """Create the Chroma vector store.

        Returns:
            Chroma: The Chroma vector store.
        """
        return VectorStoreFactory.create_chroma()

    @property
    def vector_store(self):
        """Get the vector store, creating it if necessary.

        Returns:
            object: The vector store.
        """
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store

    @property
    def retriever(self):
        """Get the retriever, creating it if necessary.

        Returns:
            object: The retriever.
        """
        if self._retriever is None:
            self._retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.k}
            )
        return self._retriever

    def load_documents_from_csv(self, csv_path=None):
        """Load documents from a CSV file and add them to the vector store.

        Args:
            csv_path (str, optional): Path to the CSV file.
                Defaults to SCHEMA_CSV_PATH from settings.

        Returns:
            RetrieverService: Self for method chaining.
        """
        csv_path = csv_path or SCHEMA_CSV_PATH
        df = pd.read_csv(csv_path)

        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=row["table_name"] + " " + row["DDL"],
                metadata={"table_name": row["table_name"]}
            )

            doc_id = str(i)
            ids.append(doc_id)
            documents.append(document)

        # Add documents to the vector store
        self.vector_store.add_documents(documents=documents, ids=ids)

        return self

    def retrieve(self, query):
        """Retrieve documents relevant to a query.

        Args:
            query (str): The query to retrieve documents for.

        Returns:
            list: The retrieved documents.
        """
        return self.retriever.invoke(query)


# Create default instance
chroma_retriever_service = RetrieverService()
