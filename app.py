import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.runnables import RunnableLambda


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\christo.jomon\\Downloads\\RAG\\api-test-459905-8577acea1327.json"


# Page config
st.set_page_config(page_title="Terrabloom", layout="centered")
st.title("Terrabloom")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load environment variables
os.environ["GEMINI_API_KEY"] = "AIzaSyCKHLCrRFIlREEr37RMuqf83E0ezWxdghY"

# Database configuration
db_config = {
    "db_user": "avnadmin",
    "db_password":"AVNS_XmWUvtwBa34zV7BHTuF",
    "db_host": "pg-langchain-mikkelkhanwald1-2c4f.l.aivencloud.com",
    "db_name": "defaultdb",
    "db_port": 27107,
    "db_schema": "customersetup"
}

@st.cache_resource
def initialize_db():
    # Create database connection
    engine = create_engine(
        f"postgresql+psycopg2://{db_config['db_user']}:{db_config['db_password']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}"
    )
    return SQLDatabase(engine, schema=db_config['db_schema'])

# @st.cache_resource
# def initialize_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0
#     )

@st.cache_resource
def initialize_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    df = pd.read_csv("table_schema.csv")
    
    vector_store = Chroma(
        collection_name="table_schema",
        persist_directory="./chrome_langchain_db1",
        embedding_function=embeddings
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 2})

def setup_chain(db, llm, retriever):
    # SQL tools
    execute_query = QuerySQLDataBaseTool(db=db)
    generate_query = create_sql_query_chain(llm, db)
    
    def strip_sql_markdown(sql: str) -> str:
        return sql.strip().replace("```sql", "").replace("```", "").strip()
    
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Retrieved Schema Context: {schema}

    Answer the question first, then format the technical details as follows:


    #### Query Details

    ### SQL Query
    ```
    {query}
    ```

    ### Query Result
    ```
    {result}
    ```

    ### Schema Context
    ```
    {schema}
    ```
    </details>
    """
    )
    
    rephrase_answer = answer_prompt | llm | StrOutputParser()
    
    chain = (
        RunnablePassthrough.assign(schema=lambda x: retriever.invoke(x["question"]))
        .assign(query=generate_query)
        .assign(
            result=itemgetter("query") | 
            RunnableLambda(lambda q: execute_query.invoke(strip_sql_markdown(q)))
        )
        | rephrase_answer
    )
    
    return chain

# Initialize components
db = initialize_db()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
    # Don't pass credentials here when using API key
)
retriever = initialize_vector_store()
chain = setup_chain(db, llm, retriever)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
