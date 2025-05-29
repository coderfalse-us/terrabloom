"""LLM models for the RAG application."""

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import LLM_MODEL, LLM_TEMPERATURE, GEMINI_API_KEY


class LLMManager:
    """Manager for LLM models."""

    def __init__(self, model_name=None, temperature=None, api_key=None):
        """Initialize the LLM manager.

        Args:
            model_name (str, optional): The name of the LLM model.
                Defaults to LLM_MODEL from settings.
            temperature (float, optional): The temperature for generation.
                Defaults to LLM_TEMPERATURE from settings.
            api_key (str, optional): The API key for Google Generative AI.
                Defaults to GEMINI_API_KEY from settings.
        """
        self.model_name = model_name or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.api_key = api_key or GEMINI_API_KEY
        self._llm = None

    @property
    def llm(self):
        """Get the LLM model, creating it if necessary.

        Returns:
            ChatGoogleGenerativeAI: The LLM model.
        """
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key  # Explicitly use API key
            )
        return self._llm

    def invoke(self, prompt):
        """Invoke the LLM with a prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM response.
        """
        return self.llm.invoke(prompt)


# Create a default instance
llm_manager = LLMManager()
