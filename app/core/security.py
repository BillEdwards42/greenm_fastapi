from google.cloud import secretmanager
from app.core.config import get_settings
from app.core.logging import logger
import functools

settings = get_settings()

@functools.lru_cache()
def get_secret(secret_id: str, version_id: str = "latest") -> str:
    """
    Fetches a secret from Google Secret Manager.
    Cached so we don't hit the API on every single request.
    """
    # Local Dev Fallback: If we have it in .env, use that first (saves API calls)
    # This helps you test locally without needing Google Permissions setup
    # if os.getenv(secret_id): return os.getenv(secret_id)

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{settings.GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"

    try:
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        return payload
    except Exception as e:
        logger.critical(f"FATAL: Could not access secret {secret_id}. Error: {e}")
        raise e