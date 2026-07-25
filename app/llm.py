"""
LLM factory — lazy-loaded, provider-agnostic.

Supports Groq (primary) with fallback. All heavy imports deferred.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache
def get_llm():
    """Get LLM instance. Lazy-loads provider libraries."""
    from app.config import get_settings
    settings = get_settings()

    # OpenRouter (free models)
    if settings.openrouter_api_key:
        try:
            from langchain_openai import ChatOpenAI
            model = settings.llm_model
            # Auto-append :free suffix if not present
            if settings.openrouter_api_key and not model.endswith(":free") and ":" not in model:
                model = f"{model}:free"
            logger.info(f"Using OpenRouter model: {model}")
            return ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model,
                temperature=0.3,
                max_tokens=4096,
            )
        except ImportError:
            logger.warning("langchain-openai not installed for OpenRouter")

    # Groq
    if settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            logger.info(f"Using Groq model: {settings.llm_model}")
            return ChatGroq(
                groq_api_key=settings.groq_api_key,
                model=settings.llm_model,
                temperature=0.3,
                max_tokens=4096,
            )
        except ImportError:
            logger.warning("langchain-groq not installed")

    raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or OPENROUTER_API_KEY.")
