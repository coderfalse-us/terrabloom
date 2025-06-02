from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from sqlalchemy import create_engine
from config.config import config

class DatabaseManager:
    """Database connection and query management"""
    
    def __init__(self):
        self.engine = None
        self.db = None
        self.query_tool = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            # Create the engine
            self.engine = create_engine(config.get_db_url())
            
            # Create the SQLDatabase object with schema
            self.db = SQLDatabase(self.engine, schema=config.DB_SCHEMA)
            
            # Create query tool
            self.query_tool = QuerySQLDataBaseTool(db=self.db)
            
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            raise
    
    def execute_query(self, query: str) -> str:
        """Execute SQL query safely"""
        try:
            cleaned_query = self._strip_sql_markdown(query)
            return self.query_tool.invoke(cleaned_query)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _strip_sql_markdown(self, sql: str) -> str:
        """Remove markdown formatting from SQL"""
        return sql.strip().replace("```sql", "").replace("```", "").strip()
    
    def get_db(self):
        """Get database instance"""
        return self.db
    
    def get_query_tool(self):
        """Get query tool instance"""
        return self.query_tool

# Global database manager instance
db_manager = DatabaseManager()