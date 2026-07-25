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

    if settings.llm_provider == "groq" and settings.groq_api_key:
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
            logger.warning("langchain-groq not installed, trying OpenAI")

    from langchain_openai import ChatOpenAI
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        logger.info("Using OpenAI fallback")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or OPENAI_API_KEY.")
