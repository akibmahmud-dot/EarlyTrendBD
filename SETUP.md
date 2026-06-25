# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- Git

## Step 1: Clone Repository

```bash
git clone https://github.com/akibmahmud-dot/EarlyTrendBD.git
cd EarlyTrendBD
```

## Step 2: Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Setup Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings (optional, defaults work fine)
```

## Step 5: Initialize Database

```bash
python db/migrations.py
```

## Step 6: Train Models (Optional)

```bash
python ml/train.py
```

This will:
- Generate 150 synthetic products
- Engineer 21 advanced features
- Train XGBoost model (~2 min)
- Train LSTM model (~3 min)
- Create ensemble model
- Save all models to `models/trained_models/`

## Step 7: Start API Server

```bash
python -m uvicorn api.app:app --reload
```

API will be available at:
- **Base URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **ReDoc:** http://localhost:8000/redoc

## Using Docker (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000
```

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov
```

## Testing API Endpoints

Using cURL:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24",
    "category": "electronics",
    "initial_sales": 500,
    "sentiment_score": 0.85,
    "social_mentions": 2000,
    "instagram_posts": 150,
    "twitter_mentions": 300,
    "tiktok_videos": 500,
    "engagement_rate": 0.15,
    "conversion_rate": 0.08,
    "search_volume": 5000,
    "views": 50000,
    "clicks": 5000
  }'

# Get viral products
curl http://localhost:8000/api/v1/viral-products?top_n=10

# Get feature importance
curl http://localhost:8000/api/v1/feature-importance

# Get model metrics
curl http://localhost:8000/api/v1/model-metrics
```

## Project Structure

```
EarlyTrendBD/
├── db/                  # Database models & migrations
├── data/                # Data generation & processing
├── ml/                  # ML models & training
├── api/                 # FastAPI backend
├── tests/               # Unit & integration tests
├── logs/                # Application logs
├── models/              # Trained model files
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker image
├── docker-compose.yml   # Docker compose config
└── README.md            # Main documentation
```

## Troubleshooting

### Port 8000 Already in Use
```bash
# Use different port
python -m uvicorn api.app:app --port 8001 --reload
```

### TensorFlow/CUDA Issues
```bash
# Install CPU-only version
pip install tensorflow-cpu
```

### Permission Denied (macOS/Linux)
```bash
chmod +x venv/bin/activate
```

### Dependencies Conflict
```bash
# Create fresh virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. Read `README_DETAILED.md` for complete documentation
2. Explore API endpoints at http://localhost:8000/docs
3. Check logs in `logs/` directory
4. Run tests: `pytest tests/ -v`

## Support

For issues:
1. Check logs in `logs/` directory
2. Review error messages carefully
3. Consult documentation
4. Open GitHub issue if needed

---

✅ **All set!** Your EarlyTrendBD system is ready to predict viral products!
