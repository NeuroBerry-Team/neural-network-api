import os
from dotenv import load_dotenv
from minio import Minio
load_dotenv()

def getMinioClient():
    print(f"[MinIO] Connecting to {os.getenv('S3_HOST')}:{os.getenv('S3_PORT')}, secure={os.getenv('ENV_MODE') == 'production'}", flush=True)

    return Minio(
        endpoint=f"{os.getenv('S3_HOST')}:{os.getenv('S3_PORT')}",
        secure=(os.getenv('ENV_MODE') == "production"),
        access_key=os.getenv('S3_ACCESS_KEY'),
        secret_key=os.getenv('S3_SECRET_KEY'),
    )