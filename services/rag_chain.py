"""Main RAG chain combining components for the RAG application."""

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage

from models.llm import llm_manager
from services.query_chain import query_chain_service
from services.retriever import chroma_retriever_service


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
                Defaults to chroma_retriever_service.
        """
        self.llm = llm or llm_manager.llm
        self.query_service = query_service or query_chain_service
        self.retriever_service = retriever_service or chroma_retriever_service
        self._chain = None
        self.waiting_for_table = False
        self.last_ambiguous_question = None

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
            system_message = SystemMessage(content="You are a data analyst who provides definitive, precise answers based on SQL query results. Provide only the direct answer without any technical details, SQL queries, or schema information. Be concise and conversational.")

            # Create a template that includes the current question and the chat history
            messages = [system_message]

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
        """Build the RAG chain.

        Args:
            include_schema (bool, optional): Whether to include schema context.
                Defaults to True.
            use_chat_prompt (bool, optional): Whether to use a chat prompt.
                Defaults to False.

        Returns:
            object: The RAG chain.
        """
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
        """Invoke the RAG chain with a question.

        Args:
            question (str): The question to answer.

        Returns:
            str: The answer or a request for clarification if the question is ambiguous.
        """
        # Check if we're waiting for a table specification from a previous ambiguous question
        if self.waiting_for_table and self.last_ambiguous_question:
            # Check if the user's response contains a table name
            tables = self.query_service.db.get_usable_table_names()
            table_mentioned = next((table for table in tables if table.lower() in question.lower()), None)

            if table_mentioned:
                # Combine the previous question with the table specification
                combined_question = f"{self.last_ambiguous_question} from {table_mentioned}"

                # Reset the waiting state
                self.waiting_for_table = False
                self.last_ambiguous_question = None

                # Process the combined question
                result = self.chain.invoke({"question": combined_question})

                return result

        # If not waiting for a table or no table was mentioned, process as a new question
        # Check if the new question is ambiguous
        is_ambiguous, message = self.query_service.is_question_ambiguous(question)

        if is_ambiguous:
            # Set the waiting state
            self.waiting_for_table = True
            self.last_ambiguous_question = question

            # Return the clarification request
            return message

        # If not ambiguous, proceed with the normal chain
        result = self.chain.invoke({"question": question})

        return result


# Create default instance
rag_chain_service = RAGChainService(retriever_service=chroma_retriever_service)
