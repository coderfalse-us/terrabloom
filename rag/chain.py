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
                query=lambda x: self._generate_query_and_store(x)
            )
            .assign(
                result=itemgetter("query")))# | 
            #     RunnableLambda(lambda q: self._safe_execute_query(q))
            # )
        #     | RunnableLambda(lambda x: self._format_response(x))
        # )
    
    def _get_schema_context(self, inputs):
        """Get relevant schema context from vector store"""
        question = inputs["question"] if isinstance(inputs, dict) else inputs
        documents = self.retriever.invoke(question)
        return "\n".join([doc.page_content for doc in documents])
    
    def _generate_query_and_store(self, inputs):
        """Generate SQL query using LLM and store in chat history"""
        print("Schema----", inputs["schema"])
        print("Schema type:", type(inputs["schema"]))
        print("Schema length:", len(inputs["schema"]) if isinstance(inputs["schema"], str) else "Not a string")
        
        if not inputs["schema"]:
            print("WARNING: Empty schema context! This will likely result in poor query generation.")
            print("Question was:", inputs["question"])

        # Generate the query
        query = llm_manager.generate_sql_query(
            question=inputs["question"],
            schema=inputs["schema"],
            chat_history=inputs.get("chat_history", [])
        )
        
        # Store the query in chat history as AI response
        self.chat_history.append(AIMessage(content=query))
        
        return query
    
    def _generate_query(self, inputs):
        """Generate SQL query using LLM (original method kept for reference)"""
        print("Schema----", inputs["schema"])
        print("Schema type:", type(inputs["schema"]))
        print("Schema length:", len(inputs["schema"]) if isinstance(inputs["schema"], str) else "Not a string")
        
        if not inputs["schema"]:
            print("WARNING: Empty schema context! This will likely result in poor query generation.")
            print("Question was:", inputs["question"])

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
        try:
            # Add user question to chat history first
            self.chat_history.append(HumanMessage(content=question))
            
            # Get response from chain (query will be added to history in _generate_query_and_store)
            response = self.chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })

            # Extract SQL query and result from response
            if isinstance(response, dict):
                sql_query = response.get('query', 'No query generated')
                result = response.get('result', 'No result available')

                # Return full response for display
                formatted_response = f"Query: {sql_query}\n\nResult: {result}"
                return formatted_response
            else:
                # Fallback if response is not a dict
                response_str = str(response)
                return response_str

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"RAG Chain Error: {error_msg}")

            # Add error to chat history
            self.chat_history.append(AIMessage(content=error_msg))

            return error_msg
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
    
    def get_store_stats(self):
        """Get vector store statistics"""
        return self.retriever.get_store_stats()

# Global RAG chain instance
rag_chain = RAGChain()