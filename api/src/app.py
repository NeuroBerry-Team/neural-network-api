from flask import Flask
from flask_cors import CORS
from src.inference_routes import inference

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=['Authorization'], supports_credentials=True)

    # Register the routes blueprints
    app.register_blueprint(inference)

    # Return the app instance
    return app