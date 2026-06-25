from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True)
    category = Column(String(100), index=True)
    platform = Column(String(50))  # daraz, shopify, instagram, etc
    platform_id = Column(String(255), unique=True)
    description = Column(Text, nullable=True)
    initial_price = Column(Float, nullable=True)
    image_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    daily_metrics = relationship("DailyMetrics", back_populates="product", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="product", cascade="all, delete-orphan")

class DailyMetrics(Base):
    __tablename__ = "daily_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Sales & Views
    sales = Column(Integer, default=0)
    views = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    
    # Social Metrics
    social_mentions = Column(Integer, default=0)
    instagram_posts = Column(Integer, default=0)
    twitter_mentions = Column(Integer, default=0)
    tiktok_videos = Column(Integer, default=0)
    
    # Sentiment & Engagement
    sentiment_score = Column(Float, default=0.5)  # 0-1 scale
    engagement_rate = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    
    # Search & Trend
    search_volume = Column(Integer, default=0)
    trend_rank = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    product = relationship("Product", back_populates="daily_metrics")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    
    # Prediction Results
    virality_score = Column(Float)  # 0-1
    confidence = Column(Float)  # 0-1
    is_viral = Column(Boolean)
    
    # Model Details
    model_name = Column(String(50))  # lstm, xgboost, ensemble
    model_version = Column(String(20))
    
    # Explainability
    top_features = Column(Text)  # JSON string of important features
    recommendation = Column(Text)
    
    prediction_date = Column(DateTime, default=datetime.utcnow)
    predicted_for_days_ahead = Column(Integer, default=7)
    
    # Relationship
    product = relationship("Product", back_populates="predictions")
