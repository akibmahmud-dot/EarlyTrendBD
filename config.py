import os
from dotenv import load_dotenv

load_dotenv()

# Application Settings
APP_NAME = "EarlyTrendBD"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "ML/AI system for predicting product virality"

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "True") == "True"

# Database Settings
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./earlytrend.db"
)

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_models")
DATA_PATH = os.getenv("DATA_PATH", "data/processed")
LOG_PATH = os.getenv("LOG_PATH", "logs")

# Training Settings
NUM_PRODUCTS = int(os.getenv("NUM_PRODUCTS", 150))
DAYS_HISTORY = int(os.getenv("DAYS_HISTORY", 30))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

# Model Architecture
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 7))
XGBOOST_MAX_DEPTH = int(os.getenv("XGBOOST_MAX_DEPTH", 7))
XGBOOST_N_ESTIMATORS = int(os.getenv("XGBOOST_N_ESTIMATORS", 100))

# Ensemble Weights
LSTM_WEIGHT = float(os.getenv("LSTM_WEIGHT", 0.4))
XGBOOST_WEIGHT = float(os.getenv("XGBOOST_WEIGHT", 0.6))

# Feature Settings
FEATURE_SCALING = os.getenv("FEATURE_SCALING", "standard") == "standard"
FEATURE_COUNT = int(os.getenv("FEATURE_COUNT", 21))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create required directories
for path in [MODEL_PATH, DATA_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

print(f"""
╔════════════════════════════════════════╗
║      EarlyTrendBD Configuration        ║
╠════════════════════════════════════════╣
║ App: {APP_NAME} v{APP_VERSION:<23}║
║ API: {API_HOST}:{API_PORT:<30}║
║ Models: {MODEL_PATH:<29}║
║ Database: {os.path.basename(DATABASE_URL):<28}║
╚════════════════════════════════════════╝
""")
