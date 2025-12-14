import os
from google.cloud import firestore
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

class FirestoreRepo():
    def __init__(self):
        # 1. Unified Authentication Logic (Same as BucketRepo)
        if os.path.exists(settings.GCP_CREDENTIALS_PATH):
            logger.info("Firestore: Using local service account file.")
            self.db = firestore.Client.from_service_account_json(
                settings.GCP_CREDENTIALS_PATH
            )
        else:
            logger.info("Firestore: Using Cloud Run default identity.")
            self.db = firestore.Client(project=settings.GCP_PROJECT_ID)
            
        self.collection = self.db.collection(settings.FIRESTORE_COLLECTION)

    async def create_or_update_user(self, uid: str, data: dict):
        """Creates or updates a user document keyed by uid."""
        try:
            # Use await if using an async library, but standard google.cloud.firestore is sync.
            # In FastAPI, standard practice for sync DB calls is to run them directly 
            # or wrap in run_in_executor. For simplicity here, we keep it sync logic 
            # wrapped in async def to satisfy the interface.
            doc_ref = self.collection.document(uid)
            doc_ref.set(data, merge = True)
            logger.info(f"User {uid} saved to firestore.")
        except Exception as e:
            logger.error(f"Failed to save user {uid}: {e}")
            return e

    async def get_user(self, uid: str) -> dict | None:
        """Retrieves a user document."""
        try:
            doc_ref = self.collection.document(uid)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get user {uid}: {e}")
            return None

    async def delete_user(self, uid: str):
        """Deletes the user generated from above testing"""
        try:
            self.collection.document(uid).delete()
            logger.info(f"Deleted user {uid}.")
        except Exception as e:
            logger.error(f"Failed to delete user {uid}: {e}")
            raise e