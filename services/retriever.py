"""Vector store and retrieval functionality for the RAG application."""

import os
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
    def create_faiss(nlist=None, embedding_function=None):
        """Create a FAISS vector store.

        Args:
            nlist (int, optional): Number of clusters for IVF indexing.
                                 If None, will be automatically determined based on dataset size.
            embedding_function (object, optional): Embedding function to use.
                Defaults to GoogleGenerativeAIEmbeddings.

        Returns:
            FAISSRetriever: The FAISS retriever instance.
        """
        embedding_function = embedding_function or GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return FAISSRetriever(nlist=nlist, embeddings_model=embedding_function)


class FAISSRetriever:
    """
    FAISS-based vector store solution that retrieves table schemas
    Uses IVF-FAISS for optimized retrieval performance
    """

    def __init__(self, nlist=None, embeddings_model=None):
        """
        Initialize the FAISSRetriever with IVF-FAISS

        Args:
            nlist (int, optional): Number of clusters for IVF indexing.
                                 If None, will be automatically determined based on dataset size.
            embeddings_model (object, optional): Embedding model to use.
                                               Defaults to GoogleGenerativeAIEmbeddings.
        """
        self.nlist = nlist  # Will be set dynamically if None
        self.embeddings_model = embeddings_model or GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Use FAISS from langchain which has better error handling
        from langchain_community.vectorstores import FAISS
        self.FAISS_cls = FAISS

        # Set index paths for IVF-FAISS
        self.index_dir = "faiss_ivf_index"
        self.index_file = f"{self.index_dir}/index.faiss"
        self.pkl_file = f"{self.index_dir}/index.pkl"

        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        """Initialize the IVF-FAISS vector store"""
        from langchain_community.vectorstores import FAISS

        # Check if index exists
        if os.path.exists(self.index_file) and os.path.exists(self.pkl_file):
            try:
                # Load existing index
                print(f"Loading existing IVF-FAISS index from {self.index_dir}")
                return FAISS.load_local(self.index_dir, self.embeddings_model, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading index: {e}")
                print("Creating new index instead...")

        # Create new index
        return self._create_store()

    def _create_store(self):
        """Create IVF-FAISS vector store with table schemas only"""
        print("Creating new IVF-FAISS index...")

        # Load data
        df = pd.read_csv("./table_schema.csv")

        # Prepare documents - only table schemas
        documents = []

        for _, row in df.iterrows():
            # Create document with full schema
            doc = Document(
                page_content=f"TABLE: {row['table_name']}\n{row['DDL']}",
                metadata={"table": row['table_name']}
            )
            documents.append(doc)

        # Determine optimal number of clusters if not set
        if self.nlist is None:
            self.nlist = self._calculate_optimal_nlist(len(documents))

        print(f"Using {self.nlist} clusters for {len(documents)} documents")

        # Create IVF-FAISS index
        vector_store = self._create_ivf_faiss_store(documents)

        # Create directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        # Save the index to disk
        vector_store.save_local(self.index_dir)
        print(f"IVF-FAISS index saved to {self.index_dir}")

        return vector_store

    def _calculate_optimal_nlist(self, num_documents):
        """Calculate optimal number of clusters based on dataset size"""
        if num_documents <= 10:
            # For very small datasets, use 1-2 clusters
            return min(2, num_documents)
        elif num_documents <= 100:
            # For small datasets, use sqrt(n) but ensure it's reasonable
            return min(int(np.sqrt(num_documents)), num_documents // 2)
        elif num_documents <= 1000:
            # For medium datasets, use sqrt(n)
            return int(np.sqrt(num_documents))
        else:
            # For large datasets, use 4*sqrt(n)
            return int(4 * np.sqrt(num_documents))

    def _create_ivf_faiss_store(self, documents):
        """Create IVF-FAISS index for better performance with large datasets"""
        try:
            import faiss

            # Get embeddings for all documents
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype('float32')

            # Get dimension
            dimension = embeddings_array.shape[1]

            # For very small datasets, use regular FAISS instead
            if len(documents) < 4:
                print(f"Dataset too small ({len(documents)} documents), using regular FAISS instead of IVF")
                return self.FAISS_cls.from_documents(documents, self.embeddings_model)

            # Create IVF index
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)

            # Train the index
            print(f"Training IVF index with {len(embeddings)} vectors and {self.nlist} clusters...")
            index.train(embeddings_array)

            # Add vectors to index
            index.add(embeddings_array)

            # Create FAISS vector store with the custom index
            vector_store = self.FAISS_cls(
                embedding_function=self.embeddings_model,
                index=index,
                docstore=self._create_docstore(documents),
                index_to_docstore_id={i: str(i) for i in range(len(documents))}
            )

            return vector_store

        except Exception as e:
            print(f"Error creating IVF-FAISS index: {e}")
            print("Falling back to regular FAISS...")
            return self.FAISS_cls.from_documents(documents, self.embeddings_model)

    def _create_docstore(self, documents):
        """Create a docstore for the documents"""
        from langchain_community.docstore.in_memory import InMemoryDocstore

        docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
        return InMemoryDocstore(docstore_dict)

    def retrieve(self, query, k=1):
        """Retrieve relevant table schemas"""
        # Use the built-in retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def get_retrieval_info(self):
        """Get information about the current retrieval method"""
        return f"Using IVF-FAISS for retrieval with {self.nlist} clusters"


class RetrieverService:
    """Service for document retrieval."""

    def __init__(self, k=None, retriever_type="chroma"):
        """Initialize the retriever service.

        Args:
            k (int, optional): Number of documents to retrieve.
                Defaults to RETRIEVER_K from settings.
            retriever_type (str, optional): Type of retriever to use.
                Options: "chroma", "faiss". Defaults to "chroma".
        """
        self.k = k or RETRIEVER_K
        self.retriever_type = retriever_type
        self._vector_store = None
        self._retriever = None

    def _create_vector_store(self):
        """Create the vector store based on retriever type.

        Returns:
            object: The vector store (Chroma or FAISS).
        """
        if self.retriever_type == "faiss":
            return VectorStoreFactory.create_faiss()
        else:
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
            if self.retriever_type == "faiss":
                # For FAISS, the vector_store is already a FAISSRetriever instance
                self._retriever = self.vector_store
            else:
                # For Chroma, use the standard as_retriever method
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
        if self.retriever_type == "faiss":
            # FAISS retriever loads documents automatically during initialization
            # No need to manually load documents
            print("FAISS retriever loads documents automatically during initialization.")
            return self

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

        # Add documents to the vector store (only for Chroma)
        self.vector_store.add_documents(documents=documents, ids=ids)

        return self

    def retrieve(self, query):
        """Retrieve documents relevant to a query.

        Args:
            query (str): The query to retrieve documents for.

        Returns:
            list: The retrieved documents.
        """
        if self.retriever_type == "faiss":
            # For FAISS, use the retrieve method with k parameter
            print(self.retriever.retrieve(query, k=self.k))
            return self.retriever.retrieve(query, k=self.k)
        else:
            # For Chroma, use the standard invoke method
            return self.retriever.invoke(query)


# Create default instances
chroma_retriever_service = RetrieverService(retriever_type="chroma")
faiss_retriever_service = RetrieverService(retriever_type="faiss")
