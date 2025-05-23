"""Vector store and retrieval functionality for the RAG application."""

import os
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import (
    SCHEMA_CSV_PATH,
    RETRIEVER_K
)


class VectorStoreFactory:
    """Factory for creating vector stores."""

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
        """Create IVF-FAISS vector store - requires manual document loading"""
        print("⚠️  FAISS vector store not found. Please load documents manually using load_documents_from_csv() method.")

        # Create a minimal empty store for now
        from langchain_community.vectorstores import FAISS

        # Create a dummy document to initialize the store
        dummy_doc = Document(
            page_content="PLACEHOLDER: No documents loaded yet",
            metadata={"table": "placeholder"}
        )

        return FAISS.from_documents([dummy_doc], self.embeddings_model)

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

    def load_documents_from_csv(self, csv_path="./table_schema.csv", force_recreate=False):
        """Load documents from CSV and create/update the FAISS index.

        Args:
            csv_path (str): Path to the CSV file containing table schemas
            force_recreate (bool): Whether to force recreation of the index

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if we should recreate or if index doesn't exist
            if force_recreate or not (os.path.exists(self.index_file) and os.path.exists(self.pkl_file)):
                print(f"Loading documents from {csv_path}...")

                # Load data
                df = pd.read_csv(csv_path)

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
                print(f"✅ IVF-FAISS index created and saved to {self.index_dir}")

                # Update the vector store
                self.vector_store = vector_store

                return True
            else:
                print("FAISS index already exists. Use force_recreate=True to recreate.")
                return True

        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return False


class RetrieverService:
    """Service for document retrieval using FAISS."""

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
        """Create the FAISS vector store.

        Returns:
            FAISSRetriever: The FAISS vector store.
        """
        return VectorStoreFactory.create_faiss()

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
            FAISSRetriever: The FAISS retriever.
        """
        if self._retriever is None:
            # For FAISS, the vector_store is already a FAISSRetriever instance
            self._retriever = self.vector_store
        return self._retriever



    def retrieve(self, query):
        """Retrieve documents relevant to a query.

        Args:
            query (str): The query to retrieve documents for.

        Returns:
            list: The retrieved documents.
        """
        # For FAISS, use the retrieve method with k parameter
        return self.retriever.retrieve(query, k=self.k)

    def load_documents_from_csv(self, csv_path="./table_schema.csv", force_recreate=False):
        """Load documents from CSV file into the FAISS vector store.

        Args:
            csv_path (str): Path to the CSV file containing table schemas
            force_recreate (bool): Whether to force recreation of the index

        Returns:
            bool: True if successful, False otherwise
        """
        return self.vector_store.load_documents_from_csv(csv_path, force_recreate)


# Create default instance
faiss_retriever_service = RetrieverService()
