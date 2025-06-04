import faiss
import numpy as np
import pickle
import zlib
import os
import json
from typing import List, Dict

class IVFFAISSStore:
    """
    Memory-efficient FAISS storage using IVF (Inverted File Index)
    Optimized for schema storage and retrieval
    """
    
    def __init__(self, embedding_dim: int = 768, nlist: int = 100):
        """
        Initialize IVF FAISS store
        
        Args:
            embedding_dim: Dimension of embeddings
            nlist: Number of clusters for IVF index (higher = more precise but slower)
        """
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.index = None
        self.document_store = {}  # Compressed document storage
        self.metadata_store = {}  # Lightweight metadata
        self.id_counter = 0
        
    def _create_ivf_index(self, sample_embeddings: np.ndarray) -> faiss.Index:
        """Create IVF index optimized for memory efficiency"""
        
        # Create quantizer (the index that produces centroids)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        
        # Create IVF index with flat storage for vectors
        # This is more memory-efficient than standard FAISS
        n_clusters = min(self.nlist, len(sample_embeddings) // 10)
        index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            n_clusters
        )
        
        # Set search parameters
        # Higher values = more accurate but slower
        index.nprobe = min(20, n_clusters // 5)
        
        return index
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict], metadata: List[Dict] = None):
        """Add documents with embeddings to the IVF store"""
        
        if len(embeddings) == 0:
            return
            
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_ivf_index(embeddings)
            
            # Train the index (required for IVF)
            if not self.index.is_trained:
                self.index.train(embeddings.astype('float32'))
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store compressed documents and metadata
        for i in range(len(documents)):
            doc_id = self.id_counter + i
            
            # Compress and store document
            self.document_store[doc_id] = self._compress_document(documents[i])
            
            # Store metadata separately (no compression for fast access)
            if metadata and i < len(metadata):
                self.metadata_store[doc_id] = metadata[i]
            else:
                self.metadata_store[doc_id] = {}
                
        # Update counter
        self.id_counter += len(documents)
    
    def _compress_document(self, document: Dict) -> bytes:
        """Compress document content"""
        # Convert to JSON and compress
        doc_json = json.dumps(document, separators=(',', ':'))  # Minimal JSON
        compressed = zlib.compress(doc_json.encode('utf-8'), level=9)
        return compressed
    
    def _decompress_document(self, compressed_doc: bytes) -> Dict:
        """Decompress document content"""
        doc_json = zlib.decompress(compressed_doc).decode('utf-8')
        return json.loads(doc_json)
    
    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Dict]:
        """Search the IVF index for similar documents"""
        print(f"IVFFAISSStore search called with k={k}")
        print(f"Query embedding shape: {query_embedding.shape}")
        
        if self.index is None:
            print("ERROR: FAISS index is None!")
            return []
            
        if self.id_counter == 0:
            print("ERROR: No documents in store (id_counter=0)")
            return []
        
        print(f"Index ntotal: {self.index.ntotal}")
        print(f"Document store size: {len(self.document_store)}")
        print(f"Metadata store size: {len(self.metadata_store)}")
            
        try:
            # Search the IVF index
            print("Executing FAISS search...")
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), k
            )
            
            print(f"Search returned {len(indices[0])} indices")
            print(f"Indices: {indices[0]}")
            print(f"Distances: {distances[0]}")
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < self.id_counter:  # Valid result
                    print(f"Processing valid result idx={idx}")
                    # Get metadata (always available)
                    metadata = self.metadata_store.get(idx, {})
                    
                    # Decompress document only when needed
                    if idx in self.document_store:
                        doc = self._decompress_document(self.document_store[idx])
                        content = doc.get('content', '')
                    else:
                        print(f"WARNING: Document {idx} not found in document_store!")
                        content = f"Document {idx} not found"
                    
                    results.append({
                        'content': content,
                        'metadata': metadata,
                        'distance': float(distances[0][i]),
                        'doc_id': idx
                    })
                else:
                    print(f"Skipping invalid index {idx} (not in range 0-{self.id_counter-1})")
            
            print(f"Returning {len(results)} search results")
            return results
        except Exception as e:
            print(f"ERROR in FAISS search: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the IVF store"""
        if not self.document_store:
            return {"error": "No documents stored"}
            
        # Calculate stats
        total_docs = len(self.document_store)
        total_compressed_size = sum(len(doc) for doc in self.document_store.values())
        
        # Sample a few documents to estimate uncompressed size
        sample_size = min(10, total_docs)
        if sample_size > 0:
            sample_ids = list(self.document_store.keys())[:sample_size]
            sample_uncompressed = sum(
                len(json.dumps(self._decompress_document(self.document_store[idx])).encode('utf-8'))
                for idx in sample_ids
            )
            estimated_uncompressed = (sample_uncompressed / sample_size) * total_docs
            compression_ratio = estimated_uncompressed / total_compressed_size if total_compressed_size > 0 else 0
        else:
            estimated_uncompressed = 0
            compression_ratio = 0
            
        return {
            "document_count": total_docs,
            "compressed_size_bytes": total_compressed_size,
            "estimated_uncompressed_bytes": estimated_uncompressed,
            "compression_ratio": compression_ratio,
            "avg_document_size_bytes": total_compressed_size / total_docs if total_docs > 0 else 0,
            "embedding_dim": self.embedding_dim,
            "nlist": self.nlist
        }
    
    def save(self, directory: str):
        """Save the IVF store to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "ivf_index.faiss"))
            
        # Save document store and metadata
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.document_store, f)
            
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata_store, f)
            
        # Save configuration
        config = {
            "embedding_dim": self.embedding_dim,
            "nlist": self.nlist,
            "id_counter": self.id_counter
        }
        
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, directory: str) -> 'IVFFAISSStore':
        """Load an IVF store from disk"""
        # Load configuration
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
            
        # Create instance
        store = cls(
            embedding_dim=config["embedding_dim"],
            nlist=config["nlist"]
        )
        
        # Load the FAISS index
        store.index = faiss.read_index(os.path.join(directory, "ivf_index.faiss"))
        
        # Load document store and metadata
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            store.document_store = pickle.load(f)
            
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            store.metadata_store = pickle.load(f)
            
        # Set counter
        store.id_counter = config["id_counter"]
        
        return store