import unittest
from unittest.mock import Mock, patch, MagicMock
from database.connection import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class"""
    
    @patch('database.connection.create_engine')
    @patch('database.connection.SQLDatabase')
    @patch('database.connection.QuerySQLDataBaseTool')
    def setUp(self, mock_query_tool, mock_sql_db, mock_create_engine):
        """Set up test fixtures with mocked dependencies"""
        self.mock_engine = Mock()
        self.mock_db = Mock()
        self.mock_query_tool = Mock()
        
        mock_create_engine.return_value = self.mock_engine
        mock_sql_db.return_value = self.mock_db
        mock_query_tool.return_value = self.mock_query_tool
        
        self.db_manager = DatabaseManager()
    
    def test_initialization(self):
        """Test DatabaseManager initialization"""
        self.assertIsNotNone(self.db_manager.engine)
        self.assertIsNotNone(self.db_manager.db)
        self.assertIsNotNone(self.db_manager.query_tool)
    
    def test_execute_query_success(self):
        """Test successful query execution"""
        self.mock_query_tool.invoke.return_value = "Query result"
        result = self.db_manager.execute_query("SELECT * FROM test")
        self.assertEqual(result, "Query result")
        self.mock_query_tool.invoke.assert_called_once_with("SELECT * FROM test")
    
    def test_execute_query_with_markdown(self):
        """Test query execution with markdown formatting"""
        self.mock_query_tool.invoke.return_value = "Query result"
        query_with_markdown = "```sql\nSELECT * FROM test\n```"
        result = self.db_manager.execute_query(query_with_markdown)
        self.mock_query_tool.invoke.assert_called_once_with("SELECT * FROM test")
    
    def test_execute_query_exception(self):
        """Test query execution with exception"""
        self.mock_query_tool.invoke.side_effect = Exception("Database error")
        result = self.db_manager.execute_query("SELECT * FROM test")
        self.assertIn("Error executing query", result)
        self.assertIn("Database error", result)
    
    def test_strip_sql_markdown(self):
        """Test SQL markdown stripping"""
        query_with_markdown = "```sql\nSELECT * FROM test\n```"
        cleaned = self.db_manager._strip_sql_markdown(query_with_markdown)
        self.assertEqual(cleaned, "SELECT * FROM test")
    
    def test_get_db(self):
        """Test getting database instance"""
        db = self.db_manager.get_db()
        self.assertEqual(db, self.mock_db)
    
    def test_get_query_tool(self):
        """Test getting query tool instance"""
        tool = self.db_manager.get_query_tool()
        self.assertEqual(tool, self.mock_query_tool)

if __name__ == '__main__':
    unittest.main()