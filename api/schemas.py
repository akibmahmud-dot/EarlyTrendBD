from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Request schemas
class PredictionRequest(BaseModel):
    """Schema for virality prediction requests"""
    product_name: str
    category: str
    initial_sales: int
    sentiment_score: float
    social_mentions: int
    instagram_posts: int = 0
    twitter_mentions: int = 0
    tiktok_videos: int = 0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0
    search_volume: int = 0
    views: int = 0
    clicks: int = 0

# Response schemas
class PredictionResponse(BaseModel):
    """Schema for prediction responses"""
    product_name: str
    virality_score: float
    confidence: float
    is_viral: bool
    recommendation: str
    top_features: List[dict]
    model_used: str
    prediction_timestamp: datetime

class TrainingResponse(BaseModel):
    """Schema for training responses"""
    status: str
    message: str
    metrics: dict
    timestamp: datetime

class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    message: str
    version: str

class AnalysisResponse(BaseModel):
    """Schema for product analysis"""
    product_name: str
    category: str
    virality_score: float
    is_viral: bool
    confidence: float
    feature_importance: dict
    recommendation: str
