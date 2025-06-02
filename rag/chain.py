from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from vector_store.retriever import IVFFAISSRetriever
from database.connection import db_manager
from llm.chat_models import llm_manager
from config.config import config # Import config

# LangSmith Tracing (optional, for debugging and monitoring)
if config.LANGCHAIN_TRACING_V2 == "true":
    from langsmith import Client
    client = Client(
        api_url=config.LANGCHAIN_ENDPOINT,
        api_key=config.LANGCHAIN_API_KEY
    )

class RAGChain:
    """Main RAG chain that orchestrates retrieval, generation, and execution"""
    
    def __init__(self):
        self.retriever = IVFFAISSRetriever()
        self.chat_history = []
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the RAG chain"""
        self.chain = (
            RunnablePassthrough()
            .assign(
                schema=lambda x: self._get_schema_context(x)
            )
            .assign(
                query=lambda x: self._generate_query(x)
            )
            .assign(
                result=itemgetter("query") | 
                RunnableLambda(lambda q: self._safe_execute_query(q))
            )
            | RunnableLambda(lambda x: self._format_response(x))
        )
    
    def _get_schema_context(self, inputs):
        """Get relevant schema context from vector store"""
        question = inputs["question"] if isinstance(inputs, dict) else inputs
        documents = self.retriever.invoke(question)
        return "\n".join([doc.page_content for doc in documents])
    
    def _generate_query(self, inputs):
        """Generate SQL query using LLM"""
        return llm_manager.generate_sql_query(
            question=inputs["question"],
            schema=inputs["schema"],
            chat_history=inputs.get("chat_history", [])
        )
    
    def _safe_execute_query(self, query: str) -> str:
        """Execute SQL query safely"""
        try:
            print(f"Executing query: {query}")
            return db_manager.execute_query(query)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _format_response(self, inputs):
        """Format the final response"""
        return llm_manager.format_response(
            question=inputs["question"],
            query=inputs["query"],
            result=inputs["result"],
            schema=inputs["schema"]
        )
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get a response"""
        # Add to chat history
        self.chat_history.append(HumanMessage(content=question))
        
        # Get response
        response = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Add response to chat history
        self.chat_history.append(AIMessage(content=response))
        
        return response
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
    
    def get_store_stats(self):
        """Get vector store statistics"""
        return self.retriever.get_store_stats()

# Global RAG chain instance
rag_chain = RAGChain()