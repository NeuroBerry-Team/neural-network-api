from minio import Minio
import os

def get_minio_client():
    return Minio(
        os.environ["S3_ENDPOINT"],
        access_key=os.environ["S3_ACCESS_KEY"],
        secret_key=os.environ["S3_SECRET_KEY"],
        secure=os.environ.get("S3_SECURE", "false").lower() == "true"
    )
