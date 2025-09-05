"""
Main application entry point.
Handles core setup and imports the FastAPI app from the API layer.
"""

import logging

# The app is imported from api.app and will be used by uvicorn
# This keeps main.py as a simple entry point while the actual
# FastAPI application lives in the API layer
from api_layer.app import app

__all__ = ["app"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
