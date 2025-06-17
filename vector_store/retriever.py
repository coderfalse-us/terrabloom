import numpy as np
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vector_store.ivf_faiss_store import IVFFAISSStore
from config.config import config

class IVFFAISSRetriever:
    """Simplified LangChain-compatible retriever for IVF FAISS store focused on table comments matching"""
    
    def __init__(self, store: IVFFAISSStore = None):
        self.store = store
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        
        # Load existing store if available
        if store is None:
            self.store = self._load_or_create_store()
    
    def _load_or_create_store(self) -> IVFFAISSStore:
        """Load existing store or create new one"""
        print(f"Attempting to load schema store from: {config.SCHEMA_STORE_PATH}")
        try:
            # Try to load existing store
            store = IVFFAISSStore.load(config.SCHEMA_STORE_PATH)
            print(f"Successfully loaded schema store with {store.id_counter} documents")
            print(f"Index size: {store.index.ntotal if store.index else 'No index'}")
            print(f"Document store size: {len(store.document_store)}")
            print(f"Metadata store size: {len(store.metadata_store)}")
            
            return store
        except Exception as e:
            print(f"ERROR loading schema store: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("Creating new schema store...")
            return self._create_schema_store()
    
    def _create_schema_store(self) -> IVFFAISSStore:
        """Create an IVF FAISS store from schema data focused on table comments"""
        # Load schema data
        df = pd.read_csv(config.TABLE_SCHEMA_PATH)
        
        documents = []
        metadata = []
        
        for i, row in df.iterrows():
            # Extract schema and table name
            schema_name, table_name = row['table_name'].split('.') if '.' in row['table_name'] else ('public', row['table_name'])
            
            # Get table comments - this is the primary content for similarity matching
            table_comments = row.get('comments', '').strip()
            
            if not table_comments:
                print(f"WARNING: No comments found for table {row['table_name']}")
                table_comments = f"Table {row['table_name']} - no description available"
            
            # Create comments-focused document for embedding and similarity matching
            comments_content = f"""TABLE_DESCRIPTION: {table_comments}
BUSINESS_PURPOSE: {table_comments}
FUNCTIONALITY: {table_comments}
USE_CASE: {table_comments}
CONTEXT: {table_comments}
TABLE_NAME: {row['table_name']}"""
            
            comments_doc = {
                'content': comments_content,
                'type': 'table_comments'
            }
            
            comments_meta = {
                'table': row['table_name'],
                'schema': schema_name,
                'table_name': table_name,
                'type': 'table_comments',
                'comments': table_comments,
                'ddl': row['DDL']  # Store DDL for final retrieval
            }
            
            documents.append(comments_doc)
            metadata.append(comments_meta)
        
        print(f"Created {len(documents)} comment-focused documents for embedding")
        
        # Get embeddings for all documents
        texts = [doc['content'] for doc in documents]
        embeddings = np.array(self.embeddings_model.embed_documents(texts))
        
        # Create IVF store
        store = IVFFAISSStore(embedding_dim=embeddings.shape[1], nlist=min(50, len(documents)//2))
        store.add_documents(embeddings, documents, metadata)
        
        # Save the store
        store.save(config.SCHEMA_STORE_PATH)
        print(f"Schema store saved to {config.SCHEMA_STORE_PATH}")
        
        return store
    
    def invoke(self, query: str, k: int = 5) -> List[Document]:
        """Simplified invoke method that matches query against table comments and returns table schemas"""
        print(f"\n=== IVFFAISSRetriever invoke called with query: '{query}' and k={k} ===")
        
        try:
            # Get query embedding
            print("Embedding query...")
            query_embedding = np.array(self.embeddings_model.embed_query(query))
            print(f"Query embedding shape: {len(query_embedding)}")
            
            # Search for similar table comments
            print("Searching for tables with similar comments...")
            results = self.store.search(query_embedding, k=k)
            print(f"Search returned {len(results)} results")
            
            if not results:
                print("WARNING: No results returned from search!")
                return []
            
            # Convert results to LangChain Documents with table schemas
            documents = []
            
            for result in results:
                metadata = result['metadata']
                
                # Create document with table schema as content
                table_schema_content = f"""TABLE: {metadata['table']}
SCHEMA: {metadata['schema']}
TABLE_NAME: {metadata['table_name']}
COMMENTS: {metadata['comments']}

FULL_DDL:
{metadata['ddl']}"""
                
                # Create LangChain Document with schema content
                doc = Document(
                    page_content=table_schema_content,
                    metadata={
                        'table': metadata['table'],
                        'schema': metadata['schema'],
                        'table_name': metadata['table_name'],
                        'comments': metadata['comments'],
                        'type': 'table_schema',
                        'similarity_score': result.get('distance', 0)  # Lower distance = higher similarity
                    }
                )
                
                documents.append(doc)
                
                print(f"Retrieved table: {metadata['table']} (similarity: {result.get('distance', 'N/A')})")
                print(f"Comments: {metadata['comments'][:100]}...")
            
            print(f"Returning {len(documents)} table schema documents")
            return documents
            
        except Exception as e:
            print(f"ERROR in invoke: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def get_store_stats(self):
        """Get statistics about the vector store"""
        return self.store.get_stats()