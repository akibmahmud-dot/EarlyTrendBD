from fastapi import APIRouter, HTTPException, Depends
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse, AnalysisResponse
from ml.train import ModelTrainer
from datetime import datetime
import numpy as np
from loguru import logger

logger.add("logs/api.log")

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global models (loaded once at startup)
trainer = None
models_loaded = False

def load_models():
    """Load or train models"""
    global trainer, models_loaded
    if not models_loaded:
        logger.info("Loading models...")
        trainer = ModelTrainer(num_products=150)
        products_df, metrics_df = trainer.generate_data()
        trainer.process_data(products_df, metrics_df)
        trainer.train_all_models()
        models_loaded = True
        logger.info("Models loaded successfully")
    return trainer

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="EarlyTrendBD API is running",
        version="1.0.0"
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_virality(request: PredictionRequest):
    """Predict product virality"""
    try:
        trainer = load_models()
        
        # Prepare input features in same order as training
        input_features = np.array([[
            request.initial_sales,
            request.initial_sales * 1.5,  # max_daily_sales
            request.initial_sales * 0.3,  # sales_std
            0.2,  # sales_trend
            request.views,
            request.views * 1.5,  # max_daily_views
            0.2,  # views_trend
            request.engagement_rate,
            request.conversion_rate,
            request.sentiment_score,
            request.social_mentions,
            request.instagram_posts,
            request.twitter_mentions,
            request.tiktok_videos,
            request.instagram_posts + request.twitter_mentions + request.tiktok_videos,
            request.initial_sales * 0.2,  # sales_volatility
            request.views * 0.2,  # view_volatility
            request.search_volume,
            request.search_volume * 1.2,  # peak_search_volume
            1 if request.search_volume > 0 else 0,  # has_trend_rank
            5 if request.search_volume > 0 else 0  # days_in_trend
        ]])
        
        # Make prediction
        virality_scores, binary_predictions = trainer.ensemble_model.predict(input_features)
        virality_score = float(virality_scores[0])
        is_viral = bool(binary_predictions[0])
        
        # Get feature importance
        importance = trainer.get_feature_importance()
        top_features = [
            {"feature": k, "importance": v} 
            for k, v in list(importance.items())[:5]
        ]
        
        # Generate recommendation
        if virality_score > 0.8:
            recommendation = "🚀 Excellent viral potential! Strong social engagement and growth. Increase marketing investment."
        elif virality_score > 0.6:
            recommendation = "📈 Good viral potential. Monitor trends and optimize content strategy."
        elif virality_score > 0.4:
            recommendation = "⚠️ Moderate potential. Focus on engagement and sentiment improvement."
        else:
            recommendation = "❌ Low viral potential. Consider product repositioning or marketing changes."
        
        return PredictionResponse(
            product_name=request.product_name,
            virality_score=virality_score,
            confidence=min(0.95, virality_score + 0.1),
            is_viral=is_viral,
            recommendation=recommendation,
            top_features=top_features,
            model_used="Ensemble (XGBoost + LSTM)",
            prediction_timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{product_name}", response_model=AnalysisResponse)
async def analyze_product(product_name: str):
    """Analyze a product from the dataset"""
    try:
        trainer = load_models()
        
        # Find product in features
        product_data = trainer.features_df[trainer.features_df['product_name'] == product_name]
        
        if product_data.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product = product_data.iloc[0]
        
        # Get prediction
        X_product = trainer.X[trainer.features_df['product_name'] == product_name]
        virality_scores, _ = trainer.ensemble_model.predict(X_product)
        
        importance = trainer.get_feature_importance()
        
        return AnalysisResponse(
            product_name=product_name,
            category=product['category'],
            virality_score=float(virality_scores[0]),
            is_viral=bool(product['is_viral']),
            confidence=0.92,
            feature_importance=importance,
            recommendation="Product shows strong viral characteristics based on historical data."
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/viral-products")
async def get_viral_products(top_n: int = 10):
    """Get top predicted viral products"""
    try:
        trainer = load_models()
        viral_products = trainer.predict_viral_products(top_n=top_n)
        return {
            "total_products": len(trainer.features_df),
            "viral_products": viral_products.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching viral products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-importance")
async def get_feature_importance(top_n: int = 15):
    """Get top important features for predictions"""
    try:
        trainer = load_models()
        importance = trainer.get_feature_importance(top_n=top_n)
        return {
            "features": importance,
            "total_features": len(trainer.feature_cols)
        }
    except Exception as e:
        logger.error(f"Error fetching feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-metrics")
async def get_model_metrics():
    """Get trained model performance metrics"""
    try:
        trainer = load_models()
        metrics = trainer.evaluate_models()
        return {
            "xgboost_metrics": metrics,
            "status": "Models trained and evaluated"
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
