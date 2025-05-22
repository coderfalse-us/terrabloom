"""Configuration settings for the RAG application."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Convert paths to strings for compatibility
BASE_DIR_STR = str(BASE_DIR)
DATA_DIR_STR = str(DATA_DIR)

# API Keys and credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCKHLCrRFIlREEr37RMuqf83E0ezWxdghY")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(BASE_DIR / "api-test-459905-8577acea1327.json")
)

# LangSmith configuration (for tracing)
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "True")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "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")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "pr-gargantuan-passing-22")

# Database configuration
DB_CONFIG = {
    "db_user": os.environ.get("DB_USER", "avnadmin"),
    "db_password": os.environ.get("DB_PASSWORD", "AVNS_XmWUvtwBa34zV7BHTuF"),
    "db_host": os.environ.get("DB_HOST", "pg-langchain-mikkelkhanwald1-2c4f.l.aivencloud.com"),
    "db_name": os.environ.get("DB_NAME", "defaultdb"),
    "db_port": int(os.environ.get("DB_PORT", "27107")),
    "db_schema": os.environ.get("DB_SCHEMA", "customersetup")
}

# Vector store configuration
VECTOR_STORE_DIR = BASE_DIR / "chrome_langchain_db1"
VECTOR_STORE_DIR_STR = ''
COLLECTION_NAME = "table_schema"
SCHEMA_CSV_PATH = DATA_DIR / "table_schema.csv"
SCHEMA_CSV_PATH_STR = str(SCHEMA_CSV_PATH)

# LLM configuration
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "models/text-embedding-004"

# Retriever configuration
RETRIEVER_K = 2  # Number of documents to retrieve
