import sys
from app.repository.bucket_repo import BucketRepo
from app.repository.firestore_repo import FirestoreRepo
from app.core.config import get_settings

settings = get_settings()

def test_infrastructure():
    print("--- STARTING PHASE 1 SMOKE TEST ---")

    # Setup filenames/IDs
    TEST_FILENAME = "smoke_test_connectivity.json"
    TEST_UID = "smoke_test_uid_001"

    storage = None
    db = None

    # --- TEST 1: STORAGE ---
    print("\n[1/2] Tesing Google Cloud Storage...")
    
    try:
        storage = BucketRepo()
        #1. Upload json
        storage.upload_json(TEST_FILENAME, '{"status": "alive"}')
        #2. Download json
        data = storage.download_json(TEST_FILENAME)
        if "alive" in data:
            print("Success: Write and Read confirmed.")
        else:
            raise Exception("Data mismatch.")

    except Exception as e:
        print(f"Failed with error: {e}")
        sys.exit(1)

    finally:
        #Clean up, runs even if above fails
        if storage:
            try:
                storage.delete_blob(TEST_FILENAME)
                print("Clean up, Test file deleted.")
            except Exception as e:
                print(f"WARNING, Could not clean up file: {e}")


    # --- TEST 2: FIRESTORE ---
    print("\n[2/2] Tesing Fire Store DataBase...")
    
    try:
        db = FirestoreRepo()
        #1. Firesotre write
        db.create_or_update_user(TEST_UID, {"uid": TEST_UID, "is_test": True})
        #2. Firestore read
        user = db.get_user(TEST_UID)
        if user and user["uid"] == TEST_UID:
            print("Success, FireStore Write and Read confirmed.")
        else:
            raise Exception("User not found.")

    except Exception as e:
        print(f"Failed with error: {e}")
        sys.exit(1)

    finally:
        #Clean up, runs even if above fails
        if db:
            try:
                db.delete_user(TEST_UID)
                print("Clean up, Test FireStore user deleted.")
            except Exception as e:
                print(f"WARNING, Could not delete the test user from firestore: {e}")

    print("--- All systems go ---")

if __name__ == "__main__":
    test_infrastructure()

