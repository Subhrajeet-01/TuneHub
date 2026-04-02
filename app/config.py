from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive = False,
    )

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "TuneHub"
    langchain_tracing_v2: bool = True

    #Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    #Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    #Redis
    redis_url: str = "redis://localhost:6379"

    #chromadb
    chroma_persist_dir: str = "./chroma_db"


@lru_cache()
def get_settings() -> Settings:
    return Settings()