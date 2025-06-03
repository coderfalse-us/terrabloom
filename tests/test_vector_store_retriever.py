import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from vector_store.retriever import IVFFAISSRetriever

class TestIVFFAISSRetriever(unittest.TestCase):
    """Test cases for IVFFAISSRetriever class"""
    
    @patch('vector_store.retriever.GoogleGenerativeAIEmbeddings')
    @patch('vector_store.retriever.IVFFAISSStore')
    def setUp(self, mock_store_class, mock_embeddings_class):
        """Set up test fixtures with mocked dependencies"""
        self.mock_store = Mock()
        self.mock_embeddings = Mock()
        
        mock_store_class.return_value = self.mock_store
        mock_embeddings_class.return_value = self.mock_embeddings
        
        with patch('vector_store.retriever.IVFFAISSRetriever._load_or_create_store'):
            self.retriever = IVFFAISSRetriever()
            self.retriever.store = self.mock_store
    
    def test_initialization(self):
        """Test retriever initialization"""
        self.assertIsNotNone(self.retriever.store)
        self.assertIsNotNone(self.retriever.embeddings_model)
    
    @patch('vector_store.retriever.IVFFAISSStore.load')
    def test_load_existing_store(self, mock_load):
        """Test loading existing store"""
        mock_store = Mock()
        mock_load.return_value = mock_store
        
        result = self.retriever._load_or_create_store()
        
        self.assertEqual(result, mock_store)
        mock_load.assert_called_once()
    
    @patch('vector_store.retriever.IVFFAISSStore.load')
    @patch('vector_store.retriever.IVFFAISSRetriever._create_schema_store')
    def test_load_store_fallback_to_create(self, mock_create, mock_load):
        """Test fallback to creating store when loading fails"""
        mock_load.side_effect = FileNotFoundError("Store not found")
        mock_new_store = Mock()
        mock_create.return_value = mock_new_store
        
        result = self.retriever._load_or_create_store()
        
        self.assertEqual(result, mock_new_store)
        mock_create.assert_called_once()
    
    @patch('vector_store.retriever.pd.read_csv')
    @patch('vector_store.retriever.IVFFAISSStore')
    def test_create_schema_store(self, mock_store_class, mock_read_csv):
        """Test creating schema store from CSV data"""
        # Mock CSV data
        mock_df = pd.DataFrame({
            'table_name': ['customers', 'orders'],
            'DDL': [
                'CREATE TABLE customers (id INT, name VARCHAR(50))',
                'CREATE TABLE orders (id INT, customer_id INT)'
            ]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock embeddings
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]] * 6  # 6 documents (2 tables + 4 columns)
        
        # Mock store creation
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        result = self.retriever._create_schema_store()
        
        self.assertEqual(result, mock_store)
        mock_store.add_documents.assert_called_once()
        mock_store.save.assert_called_once()
    
    def test_invoke(self):
        """Test retriever invoke method"""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Mock store search results
        mock_results = [
            {'content': 'CREATE TABLE customers', 'metadata': {'table': 'customers'}, 'distance': 0.8, 'doc_id': 1},
            {'content': 'CREATE TABLE orders', 'metadata': {'table': 'orders'}, 'distance': 0.7, 'doc_id': 2}
        ]
        self.mock_store.search.return_value = mock_results
        
        results = self.retriever.invoke("show me customer data")
        
        self.assertEqual(len(results), 2)
        self.mock_store.search.assert_called_once()
    
    def test_get_store_stats(self):
        """Test getting store statistics"""
        self.mock_store.get_stats.return_value = {"total_documents": 100}
        
        stats = self.retriever.get_store_stats()
        
        self.assertEqual(stats, {"total_documents": 100})
        self.mock_store.get_stats.assert_called_once()

if __name__ == '__main__':
    unittest.main()