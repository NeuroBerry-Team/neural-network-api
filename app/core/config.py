from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Neural Network API"
    # Add more config variables

settings = Settings()