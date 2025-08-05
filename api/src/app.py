from flask import Flask
from flask_cors import CORS
from src.inference_routes import inference
from src.inference_service.inference import InferenceService


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}}, 
         expose_headers=['Authorization'], supports_credentials=True)


    # Register the routes blueprints
    app.register_blueprint(inference)

    try:
        print("Loading AI model at startup...", flush=True)
        service = InferenceService()
        service.warmup()  # Warm up the model to ensure it's ready for predictions
        print("AI model loaded and warmed up successfully.", flush=True)
    except Exception as e:
        print(f"Error loading AI model: {e}", flush=True)

    # Return the app instance
    return app