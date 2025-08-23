from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router
from app.services.inference import InferenceService
#from app.core.logging import setup_logging

#setup_logging()

app = FastAPI(
    title="Neural Network API",
    version="1.0.0"
)

try:
    print("Loading AI model at startup...", flush=True)
    service = InferenceService()
    service.warmup()  # Warm up the model to ensure it's ready for predictions
    print("AI model loaded and warmed up successfully.", flush=True)
except Exception as e:
    print(f"Error loading AI model: {e}", flush=True)

app.include_router(api_router, prefix="/api/v1")