import os
from google.cloud import storage
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

class BucketRepo():
    def __init__(self):
        if os.path.exists(settings.GCP_CREDENTIALS_PATH):
            self.client = storage.Client.from_service_account_json(
                settings.GCP_CREDENTIALS_PATH
            )
        else:
            self.client = storage.Client(project=settings.GCP_PROJECT_ID)
            
        self.bucket = self.client.bucket(settings.GCS_BUCKET_NAME)

    async def upload_json(self, destination_blob_name: str, json_data: str):
        # Note: GCS library is blocking, but fast enough for this use case.
        # Ideally, run in threadpool, but keeping it simple is fine here.
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(json_data, content_type="application/json")
        logger.info(f"Uploaded {destination_blob_name} to GCS.") # Use logger!

    async def download_json(self, source_blob_name: str) -> str:
        """Downloads a blob as a string"""
        blob = self.bucket.blob(source_blob_name)
        return blob.download_as_text()

    async def delete_blob(self, blob_name: str):
        """Deletes the json that is uploaded from above testing."""
        blob = self.bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Deleted {blob_name}.")