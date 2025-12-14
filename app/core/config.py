from google.cloud import storage
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    #App Config
    PROJECT_NAME: str = "Green Moment Backend V2"
    LOG_LEVEL: str = "INFO"

    #GCP Config
    GCP_PROJECT_ID: str = "greenm02"
    GCP_CREDENTIALS_PATH: str = "credentials.json"

    #Resources
    GCS_BUCKET_NAME: str = "gm-intelligence-cache-greenm02-v2"
    FIRESTORE_COLLECTION: str = "users_v2"

    # Secret Names (Not the values, just the IDs in GCP)
    SECRET_CWA_KEY: str = "CWA_API_KEY"
    SECRET_BD_USER: str = "BRIGHT_DATA_USERNAME"
    SECRET_BD_PASS: str = "BRIGHT_DATA_PASSWORD"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()