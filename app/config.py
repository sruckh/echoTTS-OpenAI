import os
from typing import List, Optional
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Core
    RUNPOD_ENDPOINT: str
    RUNPOD_API_KEY: SecretStr
    
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Text Processing
    MAX_WORDS_PER_CHUNK: int = 40
    CHUNK_OVERLAP: int = 0
    
    # RunPod Interaction
    MAX_CONCURRENT_REQUESTS: int = 3
    # Timeout for HTTP connect/read to RunPod
    RUNPOD_CONNECT_TIMEOUT: float = 120.0
    # Global timeout for the entire job (polling until completion)
    RUNPOD_JOB_TIMEOUT_SECONDS: float = 300.0 
    
    # Security
    REQUIRE_AUTH: bool = False
    BRIDGE_TOKEN: Optional[SecretStr] = None
    
    # Application Logic
    RESPONSE_FORMATS: List[str] = ["mp3", "opus", "aac", "flac", "wav"]
    # Comma-separated map: "openai_voice:runpod_file.mp3,..."
    # Or path to JSON file
    VOICE_MAP: str = "alloy:EARS p004 freeform.mp3,echo:EARS p005.mp3,fable:EARS p004 freeform.mp3,onyx:EARS p005.mp3,nova:EARS p004 freeform.mp3,shimmer:EARS p005.mp3"
    VOICE_MAP_FILE: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def upper_log_level(cls, v: str) -> str:
        return v.upper()

settings = Settings()
