import unittest
from unittest.mock import Mock, patch, MagicMock
from llm.chat_models import LLMManager

class TestLLMManager(unittest.TestCase):
    """Test cases for LLMManager class"""
    
    @patch('llm.chat_models.ChatGoogleGenerativeAI')
    def setUp(self, mock_chat_ai):
        """Set up test fixtures with mocked dependencies"""
        self.mock_llm = Mock()
        mock_chat_ai.return_value = self.mock_llm
        self.llm_manager = LLMManager()
    
    def test_initialization(self):
        """Test LLMManager initialization"""
        self.assertIsNotNone(self.llm_manager.llm)
        self.assertIsNotNone(self.llm_manager.sql_prompt)
        self.assertIsNotNone(self.llm_manager.chat_prompt)
        self.assertIsNotNone(self.llm_manager.generate_query)
        self.assertIsNotNone(self.llm_manager.rephrase_answer)
    
    @patch('llm.chat_models.LLMManager._setup_chains')
    def test_generate_sql_query(self, mock_setup_chains):
        """Test SQL query generation"""
        # Mock the chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = "SELECT * FROM customers WHERE name = 'test'"
        self.llm_manager.generate_query = mock_chain
        
        result = self.llm_manager.generate_sql_query(
            question="Show me all customers named test",
            schema="CREATE TABLE customers (id INT, name VARCHAR)",
            chat_history=[]
        )
        
        self.assertEqual(result, "SELECT * FROM customers WHERE name = 'test'")
        mock_chain.invoke.assert_called_once()
    
    @patch('llm.chat_models.LLMManager._setup_chains')
    def test_format_response(self, mock_setup_chains):
        """Test response formatting"""
        # Mock the chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Found 1 customer named test"
        self.llm_manager.rephrase_answer = mock_chain
        
        result = self.llm_manager.format_response(
            question="Show me all customers named test",
            query="SELECT * FROM customers WHERE name = 'test'",
            result="[(1, 'test')]",
            schema="CREATE TABLE customers (id INT, name VARCHAR)"
        )
        
        self.assertEqual(result, "Found 1 customer named test")
        mock_chain.invoke.assert_called_once()
    
    def test_get_llm(self):
        """Test getting LLM instance"""
        llm = self.llm_manager.get_llm()
        self.assertEqual(llm, self.mock_llm)

if __name__ == '__main__':
    unittest.main()