# Terrabloom RAG System

A Retrieval-Augmented Generation (RAG) system for querying SQL databases using natural language, with support for multiple vector databases.

## Project Structure

```
project/
├── config/               # Configuration settings
├── data/                 # Data files
├── models/               # Model definitions
├── services/             # Service implementations
├── utils/                # Utility functions
├── app.py                # Streamlit web interface
├── main.py               # Command-line interface
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Features

- Natural language queries to SQL database
- IVF-FAISS vector database for optimized retrieval
- Modular, maintainable code structure
- Web interface with Streamlit
- Command-line interface
- Manual vector database loading from schema

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/terrabloom.git
   cd terrabloom
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```
   export GEMINI_API_KEY="your-api-key"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   ```

## Usage

### Web Interface

Run the Streamlit app:
```
streamlit run app.py
```

### Command Line

Run a single query:
```
python main.py --question "How many accounts have USD currency?"
```

Run in interactive mode:
```
python main.py
```

## Configuration

Configuration settings are stored in `config/settings.py`. You can modify these settings or override them with environment variables.

## Vector Database

### IVF-FAISS

The application uses IVF-FAISS (Inverted File with Flat compression) for vector similarity search. This provides:

- **High Performance**: Optimized for large-scale similarity search
- **Automatic Clustering**: Dynamically determines optimal number of clusters
- **Memory Efficient**: Uses inverted file structure for faster retrieval
- **Manual Control**: Load and recreate vector database as needed

## Development

To add a new vector database:

1. Add the necessary dependencies to `requirements.txt`
2. Implement a new factory method in `services/retriever.py`
3. Create a new retriever service instance
4. Update the UI to include the new option

## License

[MIT License](LICENSE)