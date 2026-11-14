from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Verifier MUST be a different model from the generator (checked at run time).
    groq_verifier_model: str = "qwen/qwen3.8-27b"
    database_url: str = "sqlite:///./clinotrust.db"  # swap for postgresql+psycopg://... later


settings = Settings()
