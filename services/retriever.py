"""Enhanced vector store and retrieval functionality for the RAG application."""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dataclasses import dataclass

from config.settings import (
    SCHEMA_CSV_PATH,
    RETRIEVER_K
)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    k: int = 5
    similarity_threshold: float = 0.7
    max_tokens_per_doc: int = 2000
    include_metadata: bool = True


class DocumentProcessor:
    """Handles document processing and enhancement."""
    
    @staticmethod
    def create_enhanced_document(row: pd.Series) -> Document:
        """Create an enhanced document with better structure and metadata.
        
        Args:
            row: DataFrame row containing table information
            
        Returns:
            Document: Enhanced document with structured content
        """
        # Extract table information
        table_name = row['table_name']
        ddl = row['DDL']
        
        # Parse DDL to extract column information (basic parsing)
        columns_info = DocumentProcessor._extract_column_info(ddl)
        
        # Create structured content
        content_parts = [
            f"Table Name: {table_name}",
            f"Schema Definition:\n{ddl}",
        ]
        
        if columns_info:
            content_parts.append(f"Columns: {', '.join(columns_info)}")
        
        page_content = "\n\n".join(content_parts)
        
        # Enhanced metadata
        metadata = {
            "table": table_name,
            "type": "schema",
            "column_count": len(columns_info),
            "columns": columns_info,
            "content_length": len(page_content)
        }
        
        return Document(page_content=page_content, metadata=metadata)
    
    @staticmethod
    def _extract_column_info(ddl: str) -> List[str]:
        """Extract column names from DDL (basic implementation)."""
        columns = []
        lines = ddl.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('CREATE') and not line.startswith('(') and not line.startswith(')'):
                # Basic column extraction - could be enhanced with proper SQL parsing
                if ' ' in line:
                    column_name = line.split()[0].strip('`,')
                    if column_name and not column_name.upper() in ['PRIMARY', 'FOREIGN', 'KEY', 'CONSTRAINT']:
                        columns.append(column_name)
        return columns


class FAISSRetriever:
    """Enhanced FAISS-based vector store with improved functionality."""

    def __init__(self, nlist: Optional[int] = None, embeddings_model=None, config: Optional[RetrievalConfig] = None):
        """Initialize the enhanced FAISS retriever."""
        self.nlist = nlist
        self.embeddings_model = embeddings_model or GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.config = config or RetrievalConfig()
        
        # Import FAISS from langchain
        from langchain_community.vectorstores import FAISS
        self.FAISS_cls = FAISS
        
        # Index management
        self.index_dir = "faiss_ivf_index"
        self.index_file = f"{self.index_dir}/index.faiss"
        self.pkl_file = f"{self.index_dir}/index.pkl"
        self.metadata_file = f"{self.index_dir}/metadata.json"
        
        # State tracking
        self.vector_store = None
        self._is_initialized = False
        self._document_count = 0
        
        # Initialize if index exists
        self._try_load_existing_index()

    def _try_load_existing_index(self) -> bool:
        """Try to load existing index."""
        if os.path.exists(self.index_file) and os.path.exists(self.pkl_file):
            try:
                print(f"Loading existing FAISS index from {self.index_dir}")
                self.vector_store = self.FAISS_cls.load_local(
                    self.index_dir, 
                    self.embeddings_model, 
                    allow_dangerous_deserialization=True
                )
                self._is_initialized = True
                self._load_metadata()
                return True
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                return False
        return False

    def _load_metadata(self):
        """Load metadata about the index."""
        import json
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self._document_count = metadata.get('document_count', 0)
                    self.nlist = metadata.get('nlist', self.nlist)
        except Exception as e:
            print(f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save metadata about the index."""
        import json
        try:
            metadata = {
                'document_count': self._document_count,
                'nlist': self.nlist,
                'created_at': pd.Timestamp.now().isoformat()
            }
            os.makedirs(self.index_dir, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    def is_ready(self) -> bool:
        """Check if the retriever is ready for use."""
        return self._is_initialized and self.vector_store is not None

    def _calculate_optimal_nlist(self, num_documents: int) -> int:
        """Calculate optimal number of clusters based on dataset size."""
        if num_documents <= 4:
            return max(1, num_documents // 2)
        elif num_documents <= 100:
            return max(2, int(np.sqrt(num_documents)))
        elif num_documents <= 1000:
            return int(np.sqrt(num_documents))
        else:
            return int(4 * np.sqrt(num_documents))

    def load_documents_from_csv(self, csv_path: str = "./table_schema.csv", force_recreate: bool = False) -> bool:
        """Load documents from CSV with enhanced processing."""
        try:
            # Check if recreation is needed
            if not force_recreate and self.is_ready():
                print("FAISS index already loaded and ready.")
                return True

            print(f"Loading documents from {csv_path}...")
            
            # Load and validate data
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            df = pd.read_csv(csv_path)
            
            if df.empty:
                raise ValueError("CSV file is empty")
                
            if 'table_name' not in df.columns or 'DDL' not in df.columns:
                raise ValueError("CSV must contain 'table_name' and 'DDL' columns")

            # Process documents with enhanced metadata
            documents = []
            for _, row in df.iterrows():
                doc = DocumentProcessor.create_enhanced_document(row)
                documents.append(doc)

            self._document_count = len(documents)
            
            # Calculate optimal clusters
            if self.nlist is None:
                self.nlist = self._calculate_optimal_nlist(len(documents))

            print(f"Creating FAISS index with {len(documents)} documents and {self.nlist} clusters")

            # Create vector store
            if len(documents) >= 4:
                self.vector_store = self._create_ivf_faiss_store(documents)
            else:
                print("Small dataset - using regular FAISS")
                self.vector_store = self.FAISS_cls.from_documents(documents, self.embeddings_model)

            # Save to disk
            os.makedirs(self.index_dir, exist_ok=True)
            self.vector_store.save_local(self.index_dir)
            self._save_metadata()
            
            self._is_initialized = True
            print(f"✅ FAISS index created and saved to {self.index_dir}")
            
            return True

        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return False

    def _create_ivf_faiss_store(self, documents: List[Document]):
        """Create IVF-FAISS index with error handling."""
        try:
            import faiss
            
            # Get embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings, dtype='float32')
            
            dimension = embeddings_array.shape[1]
            
            # Create IVF index
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            
            # Train and add vectors
            print("Training IVF index...")
            index.train(embeddings_array)
            index.add(embeddings_array)
            
            # Create docstore
            from langchain_community.docstore.in_memory import InMemoryDocstore
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
            
            # Create FAISS vector store
            vector_store = self.FAISS_cls(
                embedding_function=self.embeddings_model,
                index=index,
                docstore=docstore,
                index_to_docstore_id={i: str(i) for i in range(len(documents))}
            )
            
            return vector_store
            
        except Exception as e:
            print(f"IVF-FAISS creation failed: {e}")
            print("Falling back to regular FAISS...")
            return self.FAISS_cls.from_documents(documents, self.embeddings_model)

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        if not self.is_ready():
            raise RuntimeError("Retriever not initialized. Call load_documents_from_csv() first.")
        
        k = k or self.config.k
        
        # Use similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by similarity threshold if configured
        if self.config.similarity_threshold > 0:
            results = [(doc, score) for doc, score in results 
                      if score >= self.config.similarity_threshold]
        
        return results

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Standard retrieve method returning documents only."""
        results_with_scores = self.retrieve_with_scores(query, k)
        return [doc for doc, _ in results_with_scores]

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "is_initialized": self._is_initialized,
            "document_count": self._document_count,
            "nlist_clusters": self.nlist,
            "similarity_threshold": self.config.similarity_threshold,
            "retrieval_k": self.config.k
        }


class VectorStoreFactory:
    """Enhanced factory for creating vector stores."""

    @staticmethod
    def create_faiss(nlist: Optional[int] = None, 
                    embedding_function=None, 
                    config: Optional[RetrievalConfig] = None) -> FAISSRetriever:
        """Create a FAISS vector store with enhanced configuration."""
        embedding_function = embedding_function or GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return FAISSRetriever(nlist=nlist, embeddings_model=embedding_function, config=config)


class RetrieverService:
    """Enhanced service for document retrieval."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize the retriever service with configuration."""
        self.config = config or RetrievalConfig()
        self._vector_store = None

    @property
    def vector_store(self) -> FAISSRetriever:
        """Get the vector store, creating it if necessary."""
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.create_faiss(config=self.config)
        return self._vector_store

    def retrieve(self, query: str, include_scores: bool = False) -> List[Document] | List[Tuple[Document, float]]:
        """Retrieve documents with optional scores."""
        if include_scores:
            return self.vector_store.retrieve_with_scores(query, k=self.config.k)
        else:
            return self.vector_store.retrieve(query, k=self.config.k)

    def load_documents_from_csv(self, csv_path: str = "./table_schema.csv", force_recreate: bool = False) -> bool:
        """Load documents from CSV file."""
        return self.vector_store.load_documents_from_csv(csv_path, force_recreate)

    def is_ready(self) -> bool:
        """Check if the service is ready for retrieval."""
        return self.vector_store.is_ready()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.vector_store.get_stats()


# Create default instance with configuration
default_config = RetrievalConfig(k=RETRIEVER_K)
faiss_retriever_service = RetrieverService(config=default_config)