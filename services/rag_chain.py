"""Main RAG chain combining components for the RAG application."""

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from models.llm import llm_manager
from services.query_chain import query_chain_service
from services.retriever import faiss_retriever_service

import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGChainService:
    """Service for the main RAG chain."""

    def __init__(self, llm=None, query_service=None, retriever_service=None):
        """Initialize the RAG chain service.

        Args:
            llm (object, optional): The LLM to use.
                Defaults to llm_manager.llm.
            query_service (QueryChainService, optional): The query service to use.
                Defaults to query_chain_service.
            retriever_service (RetrieverService, optional): The retriever service to use.
                Defaults to faiss_retriever_service.
        """
        self.llm = llm or llm_manager.llm
        self.query_service = query_service or query_chain_service
        self.retriever_service = retriever_service or faiss_retriever_service
        self._chain = None
        self.waiting_for_table = False
        self.last_ambiguous_question = None
        self.chat_history = []  # Initialize empty chat history
        self.max_history_pairs = 1  # Store only the last Q&A pair (2 messages)

    def create_prompt(self, include_schema=True, use_chat_prompt=True):
        """Create a prompt for the RAG chain.

        Args:
            include_schema (bool, optional): Whether to include schema context.
                Defaults to True.
            use_chat_prompt (bool, optional): Whether to use a chat prompt.
                Defaults to False.

        Returns:
            object: The prompt.
        """
        if use_chat_prompt:
            system_message = SystemMessage(content="""You are a data analyst who provides definitive, precise answers based on SQL query results. 
Provide only the direct answer without any technical details, SQL queries, or schema information. Be concise and conversational.

IMPORTANT INSTRUCTIONS FOR FOLLOW-UP QUESTIONS:
1. When a user asks if something exists (e.g., "Is there a customer similar to X?") and then follows up with "What's the name?", you MUST provide the SPECIFIC names that are similar to X, not just any names.
2. For questions about similarity, provide the actual items that are similar to the referenced entity, with clear explanations of how they are similar.
3. Always maintain context between questions - if a follow-up question refers to a previous question, your answer must directly address the specific entity mentioned in the previous question.
4. When asked about names or specific values, provide the exact names or values from the database that relate to the previous question.""")
            # Create a template that includes the current question and the chat history
            messages = [system_message]

            messages.append(MessagesPlaceholder(variable_name="chat_history"))

            # Add the human message with the current question and context
            human_message = ("human", "Question: {question}\nSQL Query: {query}\nSQL Result: {result}" +
                            ("\nSchema Context: {schema}" if include_schema else ""))

            messages.append(human_message)

            return ChatPromptTemplate.from_messages(messages)
        else:
            # Return a regular prompt template for non-chat mode
            template = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
""" + ("""
Retrieved Schema Context: {schema}

Provide a clear, concise answer to the question. Do not include any technical details, SQL queries, or schema information in your response. Just give a direct answer to the question in a natural, conversational way.
""" if include_schema else "")

            return PromptTemplate.from_template(template)


    def build_chain(self, include_schema=True, use_chat_prompt=False):
        """Build the RAG chain."""
        prompt = self.create_prompt(include_schema, use_chat_prompt)
        rephrase_answer = prompt | self.llm | StrOutputParser()

        if include_schema:
            chain = (
                RunnablePassthrough.assign(
                    schema=lambda x: self.retriever_service.retrieve(x["question"])
                )
                .assign(
                    query=lambda x: self.query_service.generate(x["question"])
                )
                .assign(
                    result=itemgetter("query") |
                    RunnableLambda(lambda q: self.query_service.safe_execute(q))
                )
                | rephrase_answer
            )
        else:
            chain = (
                RunnablePassthrough.assign(
                    query=lambda x: self.query_service.generate(x["question"])
                )
                .assign(
                    result=itemgetter("query") |
                    RunnableLambda(lambda q: self.query_service.safe_execute(q))
                )
                | rephrase_answer
            )

        return chain

    @property
    def chain(self):
        """Get the RAG chain, creating it if necessary.

        Returns:
            object: The RAG chain.
        """
        if self._chain is None:
            self._chain = self.build_chain()
        return self._chain

    def invoke(self, question):
        """Invoke the RAG chain with a question."""
        # Check if the question is ambiguous
        is_ambiguous, message = self.query_service.is_question_ambiguous(question)
        if is_ambiguous:
            self.waiting_for_table = True
            self.last_ambiguous_question = question
            return message
        
        # Get only the most recent messages for context (limited to max_history_pairs)
        recent_history = self.chat_history[-2*self.max_history_pairs:] if self.chat_history else []
        
        # Process the question with the recent history
        result = self.chain.invoke({
            "question": question,
            "chat_history": recent_history
        })
        
        
        # Update chat history with this Q&A pair
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result))
        
        # Limit chat history to only the specified number of Q&A pairs
        max_messages = 2 * self.max_history_pairs
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

        print(self.chat_history)
        
        
        return result


# Create default instance
rag_chain_service = RAGChainService(retriever_service=faiss_retriever_service)
