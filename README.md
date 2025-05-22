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
- Multiple vector database support (Chroma and Qdrant)
- Modular, maintainable code structure
- Web interface with Streamlit
- Command-line interface

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

Use Qdrant instead of Chroma:
```
python main.py --vector-store qdrant
```

## Configuration

Configuration settings are stored in `config/settings.py`. You can modify these settings or override them with environment variables.

## Vector Databases

### Chroma

Chroma is the default vector database. It's lightweight and easy to use, with good performance for most use cases.

### Qdrant

Qdrant is an alternative vector database with excellent filtering capabilities. It's used in-memory by default in this project.

## Development

To add a new vector database:

1. Add the necessary dependencies to `requirements.txt`
2. Implement a new factory method in `services/retriever.py`
3. Create a new retriever service instance
4. Update the UI to include the new option

## License

[MIT License](LICENSE)