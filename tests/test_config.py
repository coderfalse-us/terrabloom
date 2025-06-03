import unittest
import os
from unittest.mock import patch
from config.config import Config

class TestConfig(unittest.TestCase):
    """Test cases for Config class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
    
    def test_config_initialization(self):
        """Test config initialization with default values"""
        self.assertIsNotNone(self.config.GEMINI_API_KEY)
        self.assertIsNotNone(self.config.DB_HOST)
        self.assertEqual(self.config.DB_PORT, 6432)
        self.assertEqual(self.config.LLM_MODEL, "gemini-2.0-flash")
        self.assertEqual(self.config.TEMPERATURE, 0)
    
    @patch.dict(os.environ, {'DB_HOST': 'test_host', 'DB_PORT': '5432'})
    def test_config_environment_variables(self):
        """Test config reads from environment variables"""
        config = Config()
        self.assertEqual(config.DB_HOST, 'test_host')
        self.assertEqual(config.DB_PORT, 5432)
    
    def test_get_db_url(self):
        """Test database URL generation"""
        url = self.config.get_db_url()
        self.assertIn('postgresql+psycopg2://', url)
        self.assertIn(self.config.DB_USER, url)
        self.assertIn(self.config.DB_HOST, url)
        self.assertIn(str(self.config.DB_PORT), url)
        self.assertIn(self.config.DB_NAME, url)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_set_environment_variables(self):
        """Test setting environment variables"""
        self.config.set_environment_variables()
        self.assertEqual(os.environ['GEMINI_API_KEY'], self.config.GEMINI_API_KEY)
        self.assertEqual(os.environ['LANGCHAIN_API_KEY'], self.config.LANGCHAIN_API_KEY)
        self.assertEqual(os.environ['LANGCHAIN_TRACING_V2'], self.config.LANGCHAIN_TRACING_V2)

if __name__ == '__main__':
    unittest.main()