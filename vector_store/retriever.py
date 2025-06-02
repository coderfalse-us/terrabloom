import numpy as np
import pandas as pd
import re
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vector_store.ivf_faiss_store import IVFFAISSStore
from config.config import config

class IVFFAISSRetriever:
    """LangChain-compatible retriever for IVF FAISS store"""
    
    def __init__(self, store: IVFFAISSStore = None):
        self.store = store
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        
        # Load existing store if available
        if store is None:
            self.store = self._load_or_create_store()
    
    def _load_or_create_store(self) -> IVFFAISSStore:
        """Load existing store or create new one"""
        try:
            # Try to load existing store
            return IVFFAISSStore.load(config.SCHEMA_STORE_PATH)
        except (FileNotFoundError, Exception):
            # Create new store if loading fails
            return self._create_schema_store()
    
    def _create_schema_store(self) -> IVFFAISSStore:
        """Create an IVF FAISS store from schema data"""
        # Load schema data
        df = pd.read_csv(config.TABLE_SCHEMA_PATH)
        
        # Prepare documents and metadata
        documents = []
        metadata = []
        
        for i, row in df.iterrows():
            # Extract column definitions from DDL
            column_pattern = r'(\w+)\s+(\w+(?:\(\d+\))?)\s+(\w+)?'
            columns = re.findall(column_pattern, row['DDL'])
            
            # Create document for table
            table_doc = {
                'content': f"TABLE: {row['table_name']} SCHEMA: {row['DDL']}",
                'type': 'table_schema'
            }
            
            table_meta = {
                'table': row['table_name'],
                'type': 'table_schema'
            }
            
            documents.append(table_doc)
            metadata.append(table_meta)
            
            # Create documents for each column
            for col in columns:
                if len(col) >= 2:
                    col_name, col_type = col[0], col[1]
                    col_doc = {
                        'content': f"TABLE: {row['table_name']} COLUMN: {col_name} TYPE: {col_type}",
                        'type': 'column_info'
                    }
                    
                    col_meta = {
                        'table': row['table_name'],
                        'column': col_name,
                        'type': 'column_info'
                    }
                    
                    documents.append(col_doc)
                    metadata.append(col_meta)
        
        # Get embeddings for all documents
        texts = [doc['content'] for doc in documents]
        embeddings = np.array(self.embeddings_model.embed_documents(texts))
        
        # Create IVF store
        store = IVFFAISSStore(embedding_dim=embeddings.shape[1], nlist=config.NLIST)
        store.add_documents(embeddings, documents, metadata)
        
        # Save the store
        store.save(config.SCHEMA_STORE_PATH)
        
        return store
    
    def invoke(self, query: str) -> List[Document]:
        """LangChain-compatible invoke method"""
        # Get query embedding
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        
        # Search
        results = self.store.search(query_embedding, k=5)
        
        # Convert to LangChain Document format
        documents = []
        for result in results:
            documents.append(Document(
                page_content=result['content'],
                metadata=result['metadata']
            ))
        
        return documents
    
    def get_store_stats(self):
        """Get statistics about the vector store"""
        return self.store.get_stats()