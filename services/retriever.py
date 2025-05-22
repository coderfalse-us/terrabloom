"""Vector store and retrieval functionality for the RAG application."""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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

    @staticmethod
    def create_qdrant(collection_name=None, embedding_function=None):
        """Create a Qdrant vector store.

        Args:
            collection_name (str, optional): Name of the collection.
                Defaults to COLLECTION_NAME from settings.
            embedding_function (object, optional): Embedding function to use.
                Defaults to embedding_manager.embeddings.

        Returns:
            Qdrant: The Qdrant vector store.
        """
        collection_name = collection_name or COLLECTION_NAME
        embedding_function = embedding_function or embedding_manager.embeddings

        # Get vector dimension from the embedding model
        sample_embedding = embedding_function.embed_query("Sample text")
        vector_size = len(sample_embedding)

        # Initialize Qdrant client
        client = QdrantClient(":memory:")  # Use in-memory storage for simplicity

        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # Create and return the Qdrant vector store with explicit embedding function
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_function  # Use 'embeddings' parameter instead of 'embedding_function'
        )


class RetrieverService:
    """Service for document retrieval."""

    def __init__(self, vector_store_type="chroma", k=None):
        """Initialize the retriever service.

        Args:
            vector_store_type (str, optional): Type of vector store to use.
                Options: "chroma", "qdrant". Defaults to "chroma".
            k (int, optional): Number of documents to retrieve.
                Defaults to RETRIEVER_K from settings.
        """
        self.vector_store_type = vector_store_type.lower()
        self.k = k or RETRIEVER_K
        self._vector_store = None
        self._retriever = None

    def _create_vector_store(self):
        """Create the vector store based on the specified type.

        Returns:
            object: The vector store.
        """
        if self.vector_store_type == "chroma":
            return VectorStoreFactory.create_chroma()
        elif self.vector_store_type == "qdrant":
            return VectorStoreFactory.create_qdrant()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

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
        import uuid

        csv_path = csv_path or SCHEMA_CSV_PATH
        df = pd.read_csv(csv_path)

        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=row["table_name"] + " " + row["DDL"],
                metadata={"table_name": row["table_name"]}
            )

            # Generate a UUID for Qdrant compatibility
            if self.vector_store_type == "qdrant":
                doc_id = str(uuid.uuid4())
            else:
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


# Create default instances
chroma_retriever_service = RetrieverService(vector_store_type="chroma")
qdrant_retriever_service = RetrieverService(vector_store_type="qdrant")
