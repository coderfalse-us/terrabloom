import unittest
from unittest.mock import Mock, patch
from rag.chain import RAGChain

class TestIntegration(unittest.TestCase):
    """Integration tests for the RAG application"""
    
    @patch('rag.chain.IVFFAISSRetriever')
    @patch('rag.chain.db_manager')
    @patch('rag.chain.llm_manager')
    def test_end_to_end_question_answering(self, mock_llm_manager, mock_db_manager, mock_retriever_class):
        """Test end-to-end question answering flow"""
        # Setup mocks
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "CREATE TABLE customers (id INT, name VARCHAR(50))"
        mock_retriever.invoke.return_value = [mock_doc]
        mock_retriever_class.return_value = mock_retriever
        
        mock_llm_manager.generate_sql_query.return_value = "SELECT * FROM customers"
        mock_db_manager.execute_query.return_value = "[(1, 'John'), (2, 'Jane')]"
        mock_llm_manager.format_response.return_value = "Found 2 customers: John and Jane"
        
        # Create RAG chain and ask question
        rag_chain = RAGChain()
        response = rag_chain.ask_question("Show me all customers")
        
        # Verify the flow
        self.assertEqual(response, "Found 2 customers: John and Jane")
        mock_retriever.invoke.assert_called_once_with("Show me all customers")
        mock_llm_manager.generate_sql_query.assert_called_once()
        mock_db_manager.execute_query.assert_called_once_with("SELECT * FROM customers")
        mock_llm_manager.format_response.assert_called_once()
    
    @patch('rag.chain.IVFFAISSRetriever')
    @patch('rag.chain.db_manager')
    @patch('rag.chain.llm_manager')
    def test_error_handling_in_chain(self, mock_llm_manager, mock_db_manager, mock_retriever_class):
        """Test error handling in the RAG chain"""
        # Setup mocks with database error
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "CREATE TABLE customers (id INT, name VARCHAR(50))"
        mock_retriever.invoke.return_value = [mock_doc]
        mock_retriever_class.return_value = mock_retriever
        
        mock_llm_manager.generate_sql_query.return_value = "SELECT * FROM customers"
        mock_db_manager.execute_query.return_value = "Error executing query: Connection failed"
        mock_llm_manager.format_response.return_value = "I encountered an error while querying the database"
        
        # Create RAG chain and ask question
        rag_chain = RAGChain()
        response = rag_chain.ask_question("Show me all customers")
        
        # Verify error handling
        self.assertIn("error", response.lower())
        mock_llm_manager.format_response.assert_called_once()

if __name__ == '__main__':
    unittest.main()