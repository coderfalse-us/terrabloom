"""SQL query generation and execution for the RAG application."""

import re
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool

from models.database import db_manager
from models.llm import llm_manager
from utils.helpers import strip_sql_markdown


class QueryChainService:
    """Service for SQL query generation and execution."""

    # List of common ambiguous terms that might appear in multiple tables
    AMBIGUOUS_TERMS = [
        "isdeleted", "deleted", "isarchived", "archived", "createdby", "modifiedby",
        "createddate", "modifieddate", "timeinepoch", "id", "customerid", "businessunitid"
    ]

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

    def is_question_ambiguous(self, question):
        """Check if a question is ambiguous (refers to columns that exist in multiple tables).

        Args:
            question (str): The question to check.

        Returns:
            tuple: (is_ambiguous, message) where is_ambiguous is a boolean and message is a string
                  explaining why the question is ambiguous (if it is).
        """
        # Convert question to lowercase for case-insensitive matching
        question_lower = question.lower()

        # Get all table names
        tables = self.db.get_usable_table_names()

        # Check if any specific table is mentioned in the question
        table_mentioned = any(table.lower() in question_lower for table in tables)

        # If a specific table is mentioned, the question is not ambiguous
        if table_mentioned:
            return False, ""

        # Simple check for common ambiguous columns
        for term in self.AMBIGUOUS_TERMS:
            if term.lower() in question_lower:
                # Simple message asking for table specification
                message = f"Please specify which table you want to query for '{term}'. Available tables: {', '.join(tables)}"
                return True, message

        # Check for general questions about data without specifying a table
        if re.search(r'\b(data|rows|records|entries|deleted)\b', question_lower):
            message = f"Please specify which table you want to query. Available tables: {', '.join(tables)}"
            return True, message

        return False, ""

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
            str: The generated SQL query or an error message if the question is ambiguous.
        """
        # Check if the question is ambiguous
        is_ambiguous, message = self.is_question_ambiguous(question)
        if is_ambiguous:
            # Return a special marker that indicates this is an ambiguous question
            return f"AMBIGUOUS_QUESTION: {message}"

        # If not ambiguous, generate the SQL query
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


query_chain_service = QueryChainService()
