"""Configuration management for the agentic API generator."""

import os
from dataclasses import dataclass
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@dataclass
class AgentConfig:
    """Configuration for the API client generation agent."""
    
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    client_name: Optional[str] = None
    async_client: bool = False
    include_models: bool = True
    include_examples: bool = True
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        settings = Settings()
        return cls(
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
