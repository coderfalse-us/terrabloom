# TerraBLOOM RAG System 🌱

A Retrieval-Augmented Generation (RAG) system for intelligent database querying using natural language. This system combines vector search with large language models to provide an intuitive interface for database interactions.

## Features

- **Natural Language to SQL**: Convert natural language questions into SQL queries
- **Vector-based Schema Retrieval**: Efficiently find relevant database schema information
- **Memory-efficient FAISS Storage**: Optimized vector storage with compression
- **Interactive Streamlit Interface**: User-friendly web application
- **Chat History**: Maintains conversation context for better responses
- **Real-time Query Execution**: Direct database integration with PostgreSQL

## Project Structure

```
terrabloom/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── table_schema.csv               # Database schema information
├── rag-final.ipynb               # Original Jupyter notebook (reference)
│
├── config/                        # Configuration management
│   ├── __init__.py
│   └── config.py                  # Application configuration
│
├── database/                      # Database connection and operations
│   ├── __init__.py
│   └── connection.py              # Database manager
│
├── vector_store/                  # Vector storage and retrieval
│   ├── __init__.py
│   ├── ivf_faiss_store.py        # IVF FAISS implementation
│   └── retriever.py              # Schema retriever
│
├── llm/                          # Language model management
│   ├── __init__.py
│   └── chat_models.py            # LLM chains and prompts
│
├── rag/                          # RAG chain orchestration
│   ├── __init__.py
│   └── chain.py                  # Main RAG chain
│
└── ivf_schema_store/             # Persisted vector store
    ├── config.json
    ├── documents.pkl
    ├── ivf_index.faiss
    └── metadata.pkl
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "c:\Users\christo.jomon\OneDrive - GEODIS\Documents\terrabloom"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional):**
   Create a `.env` file or set environment variables:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   GEMINI_API_KEY=your_gemini_api_key
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_HOST=your_db_host
   DB_NAME=your_db_name
   DB_PORT=your_db_port
   DB_SCHEMA=your_db_schema
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`

### Using the Interface

1. **Ask Questions**: Type natural language questions about your database
2. **View Responses**: Get formatted answers with underlying SQL queries
3. **Chat History**: Previous conversations are maintained for context
4. **System Stats**: Monitor vector store performance in the sidebar

### Example Questions

- "How many customers do we have?"
- "Show me all containers with weight capacity over 1000"
- "What are the different types of locations?"
- "Find all deleted records in the system"

## Architecture

### Components

1. **Configuration Layer** (`config/`)
   - Centralized configuration management
   - Environment variable handling
   - Database connection parameters

2. **Database Layer** (`database/`)
   - PostgreSQL connection management
   - Safe query execution
   - Error handling

3. **Vector Store Layer** (`vector_store/`)
   - IVF FAISS implementation for efficient similarity search
   - Document compression and storage
   - Schema-aware retrieval

4. **LLM Layer** (`llm/`)
   - Google Gemini integration
   - Prompt templates for SQL generation
   - Response formatting

5. **RAG Orchestration** (`rag/`)
   - Coordinates all components
   - Manages conversation flow
   - Handles query execution pipeline

6. **User Interface** (`app.py`)
   - Streamlit-based web interface
   - Real-time chat functionality
   - System monitoring dashboard

### Data Flow

1. **User Input**: Natural language question
2. **Schema Retrieval**: Find relevant database schema using vector search
3. **SQL Generation**: Convert question + schema to SQL using LLM
4. **Query Execution**: Run SQL against database
5. **Response Formatting**: Format results into natural language
6. **Display**: Show formatted response to user

## Configuration

The system uses a centralized configuration approach. Key settings include:

- **API Keys**: Google/Gemini API credentials
- **Database**: PostgreSQL connection parameters
- **Models**: LLM and embedding model specifications
- **Vector Store**: FAISS index parameters

## Performance Features

- **Memory Efficiency**: Compressed document storage
- **Fast Retrieval**: IVF FAISS indexing
- **Caching**: Persistent vector store
- **Optimized Queries**: Schema-aware SQL generation

## Security Considerations

- API keys should be stored as environment variables
- Database credentials should not be hardcoded
- SQL injection protection through parameterized queries
- Input validation and error handling

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Connection**: Verify database credentials and connectivity
3. **API Keys**: Check Google/Gemini API key validity
4. **Vector Store**: Ensure `table_schema.csv` exists and is properly formatted

### Logs and Debugging

- SQL queries are printed to console during execution
- Streamlit provides error messages in the interface
- Check terminal output for detailed error information

## Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Update documentation for new features
4. Test thoroughly before deployment

## License

This project is part of the TerraBLOOM system for GEODIS.