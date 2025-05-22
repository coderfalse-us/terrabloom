"""SQL query generation and execution for the RAG application."""

from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser

from models.database import db_manager
from models.llm import llm_manager
from utils.helpers import strip_sql_markdown


class QueryChainService:
    """Service for SQL query generation and execution."""
    
    def __init__(self, db=None, llm=None):
        """Initialize the query chain service.
        
        Args:
            db (SQLDatabase, optional): The database to query.
                Defaults to db_manager.db.
            llm (object, optional): The LLM to use for query generation.
                Defaults to llm_manager.llm.
        """
        self.db = db or db_manager.db
        self.llm = llm or llm_manager.llm
        self._generate_query = None
        self._execute_query = None
    
    @property
    def generate_query(self):
        """Get the query generation chain, creating it if necessary.
        
        Returns:
            object: The query generation chain.
        """
        if self._generate_query is None:
            self._generate_query = create_sql_query_chain(self.llm, self.db)
        return self._generate_query
    
    @property
    def execute_query(self):
        """Get the query execution tool, creating it if necessary.
        
        Returns:
            QuerySQLDataBaseTool: The query execution tool.
        """
        if self._execute_query is None:
            self._execute_query = QuerySQLDataBaseTool(db=self.db)
        return self._execute_query
    
    def generate(self, question):
        """Generate a SQL query for a question.
        
        Args:
            question (str): The question to generate a query for.
            
        Returns:
            str: The generated SQL query.
        """
        return self.generate_query.invoke({"question": question})
    
    def execute(self, query):
        """Execute a SQL query.
        
        Args:
            query (str): The SQL query to execute.
            
        Returns:
            str: The query result.
        """
        clean_query = strip_sql_markdown(query)
        return self.execute_query.invoke(clean_query)
    
    def safe_execute(self, query):
        """Safely execute a SQL query, handling exceptions.
        
        Args:
            query (str): The SQL query to execute.
            
        Returns:
            str: The query result or error message.
        """
        try:
            return self.execute(query)
        except Exception as e:
            return f"Error executing query: {str(e)}"


# Create a default instance
query_chain_service = QueryChainService()
