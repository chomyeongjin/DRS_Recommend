"""Configuration management using pydantic-settings"""
from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_title: str = "DRS - Drawing Stock"

    # AI Features
    gemini_api_key: str = ""

    # CORS Settings
    cors_origins: str = '["http://localhost:8080", "http://127.0.0.1:8080"]'

    # Data Settings
    target_len: int = 128
    cache_ttl_sec: int = 86400  # 24 hours
    max_tickers: int = 5000

    # Rate Limiting
    rate_limit_ingest: str = "5/minute"
    rate_limit_similar: str = "20/minute"

    # Feature Settings
    min_data_points: int = 30
    min_ma_points: int = 25

    # Log Level
    log_level: str = "INFO"

    # PostgreSQL Settings
    pg_host: str = "172.17.240.1"
    pg_port: int = 5433
    pg_database: str = "drs_db"
    pg_user: str = "postgres"
    pg_password: str = ""
    pg_min_conn: int = 1
    pg_max_conn: int = 10

    # Data Source Selection
    data_source: str = "parquet"  # "parquet" or "postgresql"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string"""
        try:
            return json.loads(self.cors_origins)
        except:
            return ["http://localhost:8080"]


# Global settings instance
settings = Settings()
