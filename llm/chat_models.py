from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from config.config import config

class LLMManager:
    """Manages LLM interactions for SQL generation and response formatting"""
    
    def __init__(self):
        # Set environment variables
        config.set_environment_variables()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE
        )
        
        # Initialize prompts and chains
        self._setup_prompts()
        self._setup_chains()
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        # SQL generation prompt
        self.sql_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an SQL expert. Generate SQL queries based on the user's question and provided schema context.
Follow these rules:
1. Use only tables and columns mentioned in the schema context
2. Write clear, efficient SQL queries
3. Consider table relationships and column types from the schema"""),
            MessagesPlaceholder(variable_name="chat_history", n_messages=2),
            ("human", "Schema Context: {schema}\nQuestion: {question}")
        ])
        
        # Response formatting prompt
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Rephrase the answer to the question based on the from LLM and schema context."),
            ("human", "Question: {question}\nSQL Query: {query}\nSQL Result: {result}\nSchema Context: {schema}")
        ])
    
    def _setup_chains(self):
        """Setup LangChain chains"""
        # SQL generation chain
        self.generate_query = self.sql_prompt | self.llm | StrOutputParser()
        
        # Response formatting chain
        self.rephrase_answer = self.chat_prompt | self.llm | StrOutputParser()
    
    def generate_sql_query(self, question: str, schema: str, chat_history: list = None) -> str:
        """Generate SQL query from question and schema"""
        inputs = {
            "question": question,
            "schema": schema,
            "chat_history": chat_history or []
        }
        return self.generate_query.invoke(inputs)
    
    def format_response(self, question: str, query: str, result: str, schema: str) -> str:
        """Format the final response"""
        inputs = {
            "question": question,
            "query": query,
            "result": result,
            "schema": schema
        }
        return self.rephrase_answer.invoke(inputs)
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm

# Global LLM manager instance
llm_manager = LLMManager()