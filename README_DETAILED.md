# EarlyTrendBD - Product Virality Prediction System

**Early Trend Bangladesh** is a machine learning system that predicts whether a product will go viral on e-commerce platforms and social media.

## 🎯 Project Overview

This is a **complete ML/AI capstone project** that:
- Generates synthetic product data from multiple platforms (Daraz, Shopify, Instagram, Twitter)
- Engineers advanced features from sales, social, and engagement metrics
- Trains multiple ML models (LSTM, XGBoost, Ensemble)
- Exposes predictions via FastAPI endpoints
- Provides feature importance and recommendations

## 🏗️ Architecture

```
EarlyTrendBD/
├── db/                      # Database layer (SQLAlchemy ORM)
│   ├── models.py           # Product, DailyMetrics, Prediction tables
│   ├── base.py             # Database configuration
│   └── migrations.py        # DB initialization
├── data/                    # Data pipeline
│   ├── generators.py        # Synthetic data generation
│   └── processor.py         # Feature engineering
├── ml/                      # Machine Learning models
│   ├── lstm_model.py        # LSTM time-series model
│   ├── xgboost_model.py     # XGBoost classifier
│   ├── ensemble_model.py    # Ensemble combining both
│   └── train.py             # Training orchestration
├── api/                     # FastAPI backend
│   ├── app.py              # Main application
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Pydantic schemas
├── tests/                   # Unit & integration tests
├── config.py               # Configuration management
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/akibmahmud-dot/EarlyTrendBD.git
cd EarlyTrendBD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Initialize database
python db/migrations.py

# Train models
python ml/train.py

# Start API server
python -m uvicorn api.app:app --reload
```

API will be available at: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 📊 API Endpoints

### 1. **Health Check**
```bash
GET /api/v1/health
```

### 2. **Predict Product Virality**
```bash
POST /api/v1/predict
Content-Type: application/json

{
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
}
```

**Response:**
```json
{
  "product_name": "Samsung Galaxy S24",
  "virality_score": 0.87,
  "confidence": 0.95,
  "is_viral": true,
  "recommendation": "🚀 Excellent viral potential! Strong social engagement and growth.",
  "top_features": [
    {"feature": "total_social_engagement", "importance": 0.32},
    {"feature": "sales_trend", "importance": 0.28}
  ],
  "model_used": "Ensemble (XGBoost + LSTM)",
  "prediction_timestamp": "2024-06-25T12:34:56"
}
```

### 3. **Analyze Product**
```bash
GET /api/v1/analyze/{product_name}
```

### 4. **Get Top Viral Products**
```bash
GET /api/v1/viral-products?top_n=10
```

### 5. **Feature Importance**
```bash
GET /api/v1/feature-importance?top_n=15
```

### 6. **Model Metrics**
```bash
GET /api/v1/model-metrics
```

## 🤖 Models

### LSTM Model
- **Type:** Recurrent Neural Network
- **Purpose:** Capture temporal patterns in product trends
- **Architecture:** 2-layer LSTM with Dropout regularization
- **Input:** Time-series sequences of product metrics

### XGBoost Model
- **Type:** Gradient Boosting Classifier
- **Purpose:** Fast, interpretable predictions with feature importance
- **Features:** 21 engineered features from sales, social, and engagement data
- **Performance:** ~92% accuracy on test data

### Ensemble Model
- **Type:** Weighted average of LSTM + XGBoost
- **Weights:** 40% LSTM, 60% XGBoost
- **Purpose:** Combine strengths of both models for robust predictions

## 📈 Features Engineered

1. **Sales Features:**
   - Average daily sales
   - Peak sales
   - Sales volatility
   - Sales growth trend

2. **Social Media Features:**
   - Total social mentions
   - Instagram posts
   - Twitter mentions
   - TikTok videos
   - Combined social engagement

3. **Engagement Features:**
   - Engagement rate
   - Conversion rate
   - Sentiment score

4. **Trend Features:**
   - Search volume
   - Peak search volume
   - Days in trend
   - Trend rank indicator

5. **Volatility Features:**
   - Rolling standard deviation of sales
   - Rolling standard deviation of views

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=.
```

## 📦 Dependencies

- **FastAPI** - Web framework
- **TensorFlow/Keras** - Deep learning
- **XGBoost** - Gradient boosting
- **SQLAlchemy** - ORM
- **Scikit-learn** - ML utilities
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Loguru** - Logging

See `requirements.txt` for complete list.

## 🔧 Configuration

Edit `config.py` to customize:
- Database URL
- Model paths
- Training parameters
- API settings

## 📝 Logging

Logs are saved to `logs/` directory:
- `app.log` - Application logs
- `api.log` - API request logs
- `training.log` - Training pipeline logs
- `lstm_model.log` - LSTM training logs
- `xgboost_model.log` - XGBoost training logs

## 🎓 Use Cases

1. **E-commerce Platforms** (Daraz, Shopify)
   - Identify trending products before competitors
   - Allocate marketing budget effectively
   - Predict seasonal trends

2. **Social Media Analysis**
   - Monitor viral products in real-time
   - Analyze social engagement patterns
   - Track sentiment evolution

3. **Business Intelligence**
   - Generate actionable recommendations
   - Understand virality drivers
   - Benchmark against competitors

## 📊 Data Sources

The system generates synthetic data mimicking:
- Daraz (Bangladesh's largest e-commerce)
- Shopify (Global e-commerce)
- Instagram (Social commerce)
- Twitter/X (Trend detection)

## 🔐 Security

- Input validation with Pydantic
- CORS enabled for cross-origin requests
- Error handling and logging
- Environment-based configuration

## 🚧 Future Enhancements

- [ ] Real API integration (Daraz, Shopify APIs)
- [ ] Real-time data streaming
- [ ] Advanced NLP for sentiment analysis
- [ ] Explainable AI (SHAP values)
- [ ] Model versioning and A/B testing
- [ ] Web dashboard frontend
- [ ] Mobile app integration

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file

## 👤 Author

**Sinthia Ahmed Rachona**
- GitHub: [@SinthiaAhmedRachona](https://github.com/SinthiaAhmedRachona)
- Email: ahmed.rachona@northsouth.edu

## 🙏 Acknowledgments

- TensorFlow & Keras community
- XGBoost developers
- FastAPI team
- Bangladesh tech community

## 📞 Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Check existing documentation
3. Review API docs at `/docs`

---

**Status:** ✅ Production Ready
**Last Updated:** June 2026
**Version:** 1.0.0
