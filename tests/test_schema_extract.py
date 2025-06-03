import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from utils.schema_extract import get_table_schema, get_all_schemas, get_all_tables, extract_all_table_schemas

class TestSchemaExtract(unittest.TestCase):
    """Test cases for schema extraction utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
    
    def test_get_table_schema_success(self):
        """Test successful table schema extraction"""
        # Mock cursor responses
        self.mock_cursor.fetchall.side_effect = [
            [('id', 'int4', None, 'NO'), ('name', 'varchar', 50, 'YES')],  # columns
            [('id',)],  # primary keys
            []  # foreign keys
        ]
        
        result = get_table_schema('public', 'customers', self.mock_connection)
        
        self.assertIn('CREATE TABLE public.customers', result)
        self.assertIn('id int4 NOT NULL', result)
        self.assertIn('name varchar(50)', result)
        self.assertIn('PRIMARY KEY (id)', result)
    
    def test_get_table_schema_exception(self):
        """Test table schema extraction with exception"""
        self.mock_cursor.execute.side_effect = Exception("Database error")
        
        result = get_table_schema('public', 'customers', self.mock_connection)
        self.assertIsNone(result)
    
    def test_get_all_schemas(self):
        """Test getting all schemas"""
        self.mock_cursor.fetchall.return_value = [('public',), ('customersetup',)]
        
        schemas = get_all_schemas(self.mock_connection)
        
        self.assertEqual(schemas, ['public', 'customersetup'])
        self.mock_cursor.execute.assert_called_once()
    
    def test_get_all_tables(self):
        """Test getting all tables in a schema"""
        self.mock_cursor.fetchall.return_value = [('customers',), ('orders',)]
        
        tables = get_all_tables('public', self.mock_connection)
        
        self.assertEqual(tables, ['customers', 'orders'])
        self.mock_cursor.execute.assert_called_once_with(
            unittest.mock.ANY, ('public',)
        )
    
    @patch('utils.schema_extract.psycopg2.connect')
    @patch('utils.schema_extract.get_all_schemas')
    @patch('utils.schema_extract.get_all_tables')
    @patch('utils.schema_extract.get_table_schema')
    @patch('utils.schema_extract.pd.DataFrame.to_csv')
    @patch('utils.schema_extract.os.makedirs')
    def test_extract_all_table_schemas(self, mock_makedirs, mock_to_csv, 
                                     mock_get_table_schema, mock_get_all_tables,
                                     mock_get_all_schemas, mock_connect):
        """Test extracting all table schemas"""
        # Mock database connection
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock schema and table discovery
        mock_get_all_schemas.return_value = ['public', 'customersetup']
        mock_get_all_tables.side_effect = [['customers'], ['orders']]
        mock_get_table_schema.side_effect = [
            'CREATE TABLE public.customers (...)',
            'CREATE TABLE customersetup.orders (...)'
        ]
        
        result = extract_all_table_schemas()
        
        expected_keys = ['public.customers', 'customersetup.orders']
        self.assertEqual(list(result.keys()), expected_keys)
        self.assertIn('CREATE TABLE public.customers', result['public.customers'])
        self.assertIn('CREATE TABLE customersetup.orders', result['customersetup.orders'])

if __name__ == '__main__':
    unittest.main()