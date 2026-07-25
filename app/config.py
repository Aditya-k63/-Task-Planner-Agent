from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    api_key: str = "dev-key"
    port: int = 8000
    max_retries: int = 3
    max_tool_calls_per_task: int = 15
    workspace_dir: str = "/tmp/task-planner-workspace"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
