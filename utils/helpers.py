"""Helper functions for the RAG application."""

import os
from pathlib import Path


def strip_sql_markdown(sql):
    """Strip markdown formatting from SQL queries.

    Args:
        sql (str): The SQL query with potential markdown formatting.

    Returns:
        str: The cleaned SQL query.
    """
    return sql.strip().replace("```sql", "").replace("```", "").strip()


def ensure_directory_exists(directory_path):
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory_path (str or Path): The directory path.

    Returns:
        Path: The directory path as a Path object.
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_environment_variables():
    """Set up environment variables from config if not already set."""
    from config.settings import (
        GEMINI_API_KEY,
        GOOGLE_APPLICATION_CREDENTIALS,
        LANGSMITH_TRACING,
        LANGSMITH_ENDPOINT,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT
    )

    # Set environment variables if not already set
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Also set GOOGLE_API_KEY

    # Comment out the GOOGLE_APPLICATION_CREDENTIALS to force API key usage
    # if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

    if "LANGSMITH_TRACING" not in os.environ:
        os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING

    if "LANGSMITH_ENDPOINT" not in os.environ:
        os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT

    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
