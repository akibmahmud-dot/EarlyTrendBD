import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./earlytrendbd.db')
DATABASE_ECHO = os.getenv('DEBUG', 'False') == 'True'

# API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
DEBUG = os.getenv('DEBUG', 'False') == 'True'
API_TITLE = 'EarlyTrendBD API'
API_VERSION = '1.0.0'

# Model
MODEL_PATH = os.getenv('MODEL_PATH', './models/trained_models')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Data Collection
DATA_COLLECTION_INTERVAL = int(os.getenv('DATA_COLLECTION_INTERVAL', 3600))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 5))

# API Keys
SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY', '')
INSTAGRAM_ACCESS_TOKEN = os.getenv('INSTAGRAM_ACCESS_TOKEN', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
DARAZ_API_KEY = os.getenv('DARAZ_API_KEY', '')

# Paths
DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
