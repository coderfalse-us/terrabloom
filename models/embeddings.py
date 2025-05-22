"""Embedding models for the RAG application."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import EMBEDDING_MODEL, GEMINI_API_KEY


class EmbeddingManager:
    """Manager for embedding models."""

    def __init__(self, model_name=None, api_key=None):
        """Initialize the embedding manager.

        Args:
            model_name (str, optional): The name of the embedding model.
                Defaults to EMBEDDING_MODEL from settings.
            api_key (str, optional): The API key for Google Generative AI.
                Defaults to GEMINI_API_KEY from settings.
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.api_key = api_key or GEMINI_API_KEY
        self._embeddings = None

    @property
    def embeddings(self):
        """Get the embeddings model, creating it if necessary.

        Returns:
            GoogleGenerativeAIEmbeddings: The embeddings model.
        """
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self.api_key  # Explicitly use API key
            )
        return self._embeddings

    def embed_query(self, text):
        """Embed a query text.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, documents):
        """Embed a list of documents.

        Args:
            documents (list): The documents to embed.

        Returns:
            list: The embedding vectors.
        """
        return self.embeddings.embed_documents(documents)


# Create a default instance
embedding_manager = EmbeddingManager()
