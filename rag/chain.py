from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Optional, Dict, Any
from vector_store.retriever import IVFFAISSRetriever
from database.connection import db_manager
from llm.chat_models import llm_manager
from config.config import config # Import config
from database.conversations import conversation_store

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
        self.active_conversation_id = None
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
                result=lambda x: self._safe_execute_query(x["query"])
            )
            | RunnableLambda(lambda x: self._format_response(x))
        )
    
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
        
        # Validate inputs
        if not inputs.get("question"):
            raise ValueError("Question cannot be empty or None")
        
        if not inputs["schema"]:
            print("WARNING: Empty schema context! This will likely result in poor query generation.")
            print("Question was:", inputs["question"])

        # Generate the query
        query = llm_manager.generate_sql_query(
            question=inputs["question"],
            schema=inputs["schema"] or "",  # Ensure schema is never None
            chat_history=inputs.get("chat_history", [])
        )
          # We'll let the ask_question method add the AI response to history
        # after getting the final formatted result
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
            # Ensure we have an active conversation
            if not self.active_conversation_id:
                self.active_conversation_id = conversation_store.create_conversation("New Conversation")
            
            # Validate the question input
            if question is None or not isinstance(question, str) or question.strip() == "":
                raise ValueError("Question cannot be empty or None")
            
            # Add user question to chat history first
            self.chat_history.append(HumanMessage(content=question))
              # Get response from chain (query will be added to history in _generate_query_and_store)
            # Create a list of messages for the last 2 exchanges (4 messages or fewer)
            recent_history = self.chat_history[-4:] if len(self.chat_history) > 4 else self.chat_history
            
            response = self.chain.invoke({
                "question": question,
                "chat_history": recent_history
            })

            # Extract response content
            response_content = response if isinstance(response, str) else str(response)
              # Save the updated conversation
            self.save_current_conversation()
            
            return response_content
            
        except ValueError as ve:
            # Handle validation errors
            error_msg = f"Validation error: {str(ve)}"
            print(f"RAG Chain Error: {error_msg}")
            
            # Add error to chat history
            self.chat_history.append(AIMessage(content=error_msg))
            
            # Still save the conversation even with the error
            self.save_current_conversation()
            
            return error_msg
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"RAG Chain Error: {error_msg}")

            # Add error to chat history
            self.chat_history.append(AIMessage(content=error_msg))
            
            # Still save the conversation even with the error
            self.save_current_conversation()

            return error_msg
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        
    def get_store_stats(self):
        """Get vector store statistics"""
        return self.retriever.get_store_stats()
        
    # Conversation management methods
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation and set it as active
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            str: Conversation ID
        """
        conv_id = conversation_store.create_conversation(title)
        self.active_conversation_id = conv_id
        self.chat_history = []
        return conv_id
    
    def load_conversation(self, conv_id: str) -> bool:
        """Load a conversation from storage and set it as active
        
        Args:
            conv_id: Conversation ID to load
            
        Returns:
            bool: Success status
        """
        messages = conversation_store.load_messages(conv_id)
        if messages:
            self.chat_history = messages
            self.active_conversation_id = conv_id
            return True
        return False
    
    def save_current_conversation(self) -> bool:
        """Save the current conversation to storage
        
        Returns:
            bool: Success status
        """
        if not self.active_conversation_id:
            # Create a new conversation if none is active
            self.active_conversation_id = conversation_store.create_conversation()
        
        return conversation_store.save_messages(
            self.active_conversation_id, 
            self.chat_history
        )
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversation metadata
        
        Returns:
            List[Dict[str, Any]]: List of conversation metadata
        """
        return conversation_store.list_conversations()
    
    def update_conversation_title(self, title: str) -> bool:
        """Update the title of the current conversation
        
        Args:
            title: New title
            
        Returns:
            bool: Success status
        """
        if not self.active_conversation_id:
            return False
        
        return conversation_store.update_conversation_title(
            self.active_conversation_id, 
            title
        )
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation
        
        Args:
            conv_id: Conversation ID
            
        Returns:
            bool: Success status
        """
        # If deleting the active conversation, clear history
        if conv_id == self.active_conversation_id:
            self.active_conversation_id = None
            self.chat_history = []
        
        return conversation_store.delete_conversation(conv_id)
        
# Global RAG chain instance
rag_chain = RAGChain()
