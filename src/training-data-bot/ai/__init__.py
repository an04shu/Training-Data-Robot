"""
AI client module for text generation and LLM services.

Since Decodo is a web scraping service, this module handles
the actual AI text generation for training data creation.
"""

from .client import AIClient

__all__ = ["AIClient"]