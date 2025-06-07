from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    """Settings for the application."""

    # Environment variables
    ENVIRONMENT: str = "local"
    PROJECT_NAME: str = "fact-check-service-dg"
    API_V1_STR: str = "/api/v1"

    IMAGE_FILE_DETAIL_API_PATH: str = "/fact-checks/image-analysis-results"
    BACKEND_ENDPOINT: str = "http://focustbackend:8000/api/v1"

    # GPU settings
    IS_GPU: bool = torch.cuda.is_available()


settings = Settings()
