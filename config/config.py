import os
from typing import Dict, Any

class Config:
    """Configuration class for the RAG application"""
    
    def __init__(self):
        # API Keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBc_8Ls8yQQsgOgeMusRW3Y8jcC3EO1E_k")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCKHLCrRFIlREEr37RMuqf83E0ezWxdghY")
        self.GROQ_API_KEY= os.getenv("GROQ_API_KEY", "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")
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "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") # Add LangSmith API Key

        # LangSmith Configuration
        self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
        self.LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self.LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "terrabloom-rag")
        
        # Database Configuration
        self.DB_USER = os.getenv("DB_USER", "usr_reporting")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "Atdd5v3ecsr3p")
        self.DB_HOST = os.getenv("DB_HOST", "alspgbdvit01q.ohl.com")
        self.DB_NAME = os.getenv("DB_NAME", "vite_reporting_r_qa")
        self.DB_PORT = int(os.getenv("DB_PORT", "6432"))
        
        # Model Configuration
        self.LLM_MODEL = "llama-3.3-70b-versatile"
        self.EMBEDDING_MODEL = "models/text-embedding-004"
        self.TEMPERATURE = 0
        
        # FAISS Configuration
        self.EMBEDDING_DIM = 768
        self.NLIST = 50
        self.SCHEMA_STORE_PATH = "ivf_schema_store"
        self.TABLE_SCHEMA_PATH = "data/table_schema.csv"
        
    def get_db_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def set_environment_variables(self):
        """Set environment variables for API keys"""
        os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
        os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
        os.environ["GROQ_API_KEY"] = self.GROQ_API_KEY
        os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_TRACING_V2"] = self.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT

# Global config instance
config = Config()
config.set_environment_variables() # Ensure env vars are set at import time