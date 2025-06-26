from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from config.config import config
from langchain_openai import ChatOpenAI

class LLMManager:
    """Manages LLM interactions for SQL generation and response formatting"""
    
    def __init__(self):
        # Set environment variables
        config.set_environment_variables()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            temperature=config.TEMPERATURE,
            base_url="https://openrouter.ai/api/v1"
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
2. Write only clear, efficient SQL queries without any explanations and only one query is more than enough
3. Strictly refer to the history as there will be follow up questions
4. When user mentions "refer [table_name]" in a follow-up, JOIN that table with the previous query context using appropriate foreign keys
5. Maintain any WHERE conditions from previous queries while adding the requested table data
6.For boolean values, use '1' or '0'
7.If you have multiple tables with same column to refer or confution in table names just ask a follow up question"""),
            MessagesPlaceholder(variable_name="chat_history", n_messages=2),
            ("human", "Schema Context: {schema}\nQuestion: {question}")
        ])
        # Response formatting prompt
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Rephrase the executed SQL result in one concise line. If no data is returned, respond with 'No such details found.' If the result contains bulk data, show only the first 5 rows. Do not add extra details. If the result is ambiguous or unclear based on the schema, respond with a follow-up question instead."
 ),
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
        # Ensure chat_history is properly formatted for the prompt
        history = chat_history or []
        
        # Debug: print history to verify it's being passed correctly
        print(f"Chat history being used (length: {len(history)}):")
        for msg in history:
            print(f"  - {msg.__class__.__name__}: {msg.content[:50]}...")
        
        inputs = {
            "question": question,
            "schema": schema,
            "chat_history": history
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