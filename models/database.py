"""Database connection and utilities for the RAG application."""

from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from config.settings import DB_CONFIG


class DatabaseManager:
    """Manager for database connections and operations."""
    
    def __init__(self, config=None):
        """Initialize the database manager with configuration.
        
        Args:
            config (dict, optional): Database configuration. Defaults to DB_CONFIG.
        """
        self.config = config or DB_CONFIG
        self._db = None
    
    @property
    def connection_string(self):
        """Get the database connection string.
        
        Returns:
            str: The connection string for SQLAlchemy.
        """
        return (
            f"postgresql+psycopg2://{self.config['db_user']}:{self.config['db_password']}"
            f"@{self.config['db_host']}:{self.config['db_port']}/{self.config['db_name']}"
        )
    
    @property
    def db(self):
        """Get the SQLDatabase instance, creating it if necessary.
        
        Returns:
            SQLDatabase: The SQLDatabase instance.
        """
        if self._db is None:
            engine = create_engine(self.connection_string)
            self._db = SQLDatabase(engine, schema=self.config['db_schema'])
        return self._db
    
    def get_table_names(self):
        """Get the names of usable tables in the database.
        
        Returns:
            list: List of table names.
        """
        return self.db.get_usable_table_names()
    
    def get_table_info(self):
        """Get information about the tables in the database.
        
        Returns:
            str: Table information.
        """
        return self.db.table_info
    
    def execute_query(self, query):
        """Execute a SQL query.
        
        Args:
            query (str): The SQL query to execute.
            
        Returns:
            str: The query result.
        """
        from langchain_community.tools import QuerySQLDataBaseTool
        
        execute_query = QuerySQLDataBaseTool(db=self.db)
        return execute_query.invoke(query)


# Create a default instance
db_manager = DatabaseManager()
