from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SmartDoc FastAPI Backend"
    app_version: str = "0.2.0"

    # Comma-separated origins, e.g. "http://localhost:3000,http://127.0.0.1:5500"
    cors_allow_origins: str = "*"

    max_upload_files: int = 20
    max_upload_size_mb: int = 20

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origins(self) -> list[str]:
        raw = [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]
        return raw or ["*"]


settings = Settings()
