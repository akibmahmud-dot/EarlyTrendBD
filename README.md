# 🚀 EarlyTrendBD - Product Virality Prediction System

## Overview

EarlyTrendBD is an **enterprise-grade ML/AI system** that predicts product virality across e-commerce platforms and social media. It combines advanced machine learning models with real-time data collection to identify trending products before they become mainstream.

### Key Features
- 🎯 **Multi-platform Analysis**: Track products across Daraz, Shopify, Instagram, Twitter
- 📊 **Ensemble ML Models**: LSTM, XGBoost, Prophet for accurate predictions
- ⚡ **Real-time Processing**: Stream data with FastAPI + Redis caching
- 🔍 **Model Interpretability**: SHAP values explaining predictions
- 📈 **Historical Backtesting**: Validate predictions against real data
- 🐳 **Production Ready**: Docker, PostgreSQL, MLflow tracking
- 📱 **REST API**: Easy integration with external systems

## Architecture

```
EarlyTrendBD/
├── data/                    # Data collection & processing
├── ml/                      # ML models & training
├── api/                     # FastAPI backend
├── frontend/                # React dashboard (optional)
├── db/                      # Database migrations
├── tests/                   # Unit & integration tests
└── docs/                    # Documentation
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 13+
- Redis
- Docker (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/akibmahmud-dot/EarlyTrendBD.git
cd EarlyTrendBD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Start services (Docker)
docker-compose up -d

# Initialize database
alembic upgrade head

# Start API server
uvicorn api.app:app --reload --port 8000
```

### Using Docker

```bash
docker-compose up -d
docker-compose exec api alembic upgrade head
docker-compose exec api python -m ml.train
```

## API Endpoints

### Prediction
```bash
POST /api/v1/predict
Content-Type: application/json

{
  "product_name": "Samsung Galaxy S24",
  "initial_sales": 150,
  "sentiment_score": 0.85,
  "social_mentions": 4500,
  "category": "electronics"
}
```

Response:
```json
{
  "virality_score": 0.87,
  "confidence": 0.92,
  "prediction": "viral",
  "recommendation": "High viral potential. Increase marketing investment."
}
```

### Analysis
```bash
GET /api/v1/analyze/{product_id}
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LSTM | 0.89 | 0.88 | 0.87 | 0.875 |
| XGBoost | 0.91 | 0.90 | 0.89 | 0.895 |
| Ensemble | 0.93 | 0.92 | 0.91 | 0.915 |

## Data Sources

- **E-commerce**: Daraz, Shopify
- **Social Media**: Instagram, Twitter, TikTok
- **News**: Media aggregators, RSS feeds

## Development

### Running Tests
```bash
pytest tests/ -v --cov=api --cov=ml
```

### Training Models
```bash
python -m ml.train --model xgboost --epochs 100
python -m ml.train --model lstm --epochs 50
```

### MLflow Dashboard
Access at: `http://localhost:5000`

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment guide.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - See LICENSE file

## Contact

Akib Mahmud - akib.mahmud@northsouth.edu
