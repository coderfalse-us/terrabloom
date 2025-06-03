import unittest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from rag.chain import RAGChain

class TestRAGChain(unittest.TestCase):
    """Test cases for RAGChain class"""
    
    @patch('rag.chain.IVFFAISSRetriever')
    @patch('rag.chain.db_manager')
    @patch('rag.chain.llm_manager')
    def setUp(self, mock_llm_manager, mock_db_manager, mock_retriever_class):
        """Set up test fixtures with mocked dependencies"""
        self.mock_retriever = Mock()
        self.mock_db_manager = Mock()
        self.mock_llm_manager = Mock()
        
        mock_retriever_class.return_value = self.mock_retriever
        mock_db_manager.return_value = self.mock_db_manager
        mock_llm_manager.return_value = self.mock_llm_manager
        
        self.rag_chain = RAGChain()
    
    def test_initialization(self):
        """Test RAGChain initialization"""
        self.assertIsNotNone(self.rag_chain.retriever)
        self.assertEqual(self.rag_chain.chat_history, [])
        self.assertIsNotNone(self.rag_chain.chain)
    
    def test_get_schema_context(self):
        """Test schema context retrieval"""
        # Mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "CREATE TABLE customers"
        mock_doc2 = Mock()
        mock_doc2.page_content = "CREATE TABLE orders"
        
        self.mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        
        context = self.rag_chain._get_schema_context({"question": "test question"})
        expected = "CREATE TABLE customers\nCREATE TABLE orders"
        self.assertEqual(context, expected)
    
    @patch('rag.chain.llm_manager')
    def test_generate_query(self, mock_llm_manager):
        """Test SQL query generation"""
        mock_llm_manager.generate_sql_query.return_value = "SELECT * FROM customers"
        
        inputs = {
            "question": "Show all customers",
            "schema": "CREATE TABLE customers",
            "chat_history": []
        }
        
        query = self.rag_chain._generate_query(inputs)
        self.assertEqual(query, "SELECT * FROM customers")
        mock_llm_manager.generate_sql_query.assert_called_once()
    
    @patch('rag.chain.db_manager')
    def test_safe_execute_query_success(self, mock_db_manager):
        """Test successful query execution"""
        mock_db_manager.execute_query.return_value = "Query result"
        
        result = self.rag_chain._safe_execute_query("SELECT * FROM customers")
        self.assertEqual(result, "Query result")
    
    @patch('rag.chain.db_manager')
    def test_safe_execute_query_exception(self, mock_db_manager):
        """Test query execution with exception"""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = self.rag_chain._safe_execute_query("SELECT * FROM customers")
        self.assertIn("Error executing query", result)
        self.assertIn("Database error", result)
    
    @patch('rag.chain.llm_manager')
    def test_format_response(self, mock_llm_manager):
        """Test response formatting"""
        mock_llm_manager.format_response.return_value = "Formatted response"
        
        inputs = {
            "question": "test question",
            "query": "SELECT * FROM customers",
            "result": "query result",
            "schema": "CREATE TABLE customers"
        }
        
        response = self.rag_chain._format_response(inputs)
        self.assertEqual(response, "Formatted response")
    
    @patch('rag.chain.RAGChain._setup_chain')
    def test_ask_question(self, mock_setup_chain):
        """Test asking a question"""
        # Mock the chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer to question"
        self.rag_chain.chain = mock_chain
        
        response = self.rag_chain.ask_question("What is the total sales?")
        
        self.assertEqual(response, "Answer to question")
        self.assertEqual(len(self.rag_chain.chat_history), 2)
        self.assertIsInstance(self.rag_chain.chat_history[0], HumanMessage)
        self.assertIsInstance(self.rag_chain.chat_history[1], AIMessage)
    
    def test_clear_history(self):
        """Test clearing chat history"""
        self.rag_chain.chat_history = [HumanMessage(content="test")]
        self.rag_chain.clear_history()
        self.assertEqual(self.rag_chain.chat_history, [])
    
    def test_get_store_stats(self):
        """Test getting vector store statistics"""
        self.mock_retriever.get_store_stats.return_value = {"total_docs": 100}
        
        stats = self.rag_chain.get_store_stats()
        self.assertEqual(stats, {"total_docs": 100})
        self.mock_retriever.get_store_stats.assert_called_once()

if __name__ == '__main__':
    unittest.main()