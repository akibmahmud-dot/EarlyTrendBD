import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from data.generators import SyntheticDataGenerator
from data.processor import DataProcessor
from ml.lstm_model import LSTMViralityPredictor
from ml.xgboost_model import XGBoostViralityPredictor
from ml.ensemble_model import EnsembleViralityPredictor
from loguru import logger

logger.add("logs/training.log")

class ModelTrainer:
    """Orchestrate model training pipeline"""
    
    def __init__(self, num_products: int = 150, days_history: int = 30):
        self.num_products = num_products
        self.days_history = days_history
        self.lstm_model = None
        self.xgboost_model = None
        self.ensemble_model = None
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_cols = None
    
    def generate_data(self):
        """Generate synthetic training data"""
        logger.info(f"Generating {self.num_products} products with {self.days_history} days of data...")
        
        generator = SyntheticDataGenerator(num_products=self.num_products)
        products_df, metrics_df = generator.generate_all(days=self.days_history)
        
        logger.info(f"Generated {len(products_df)} products and {len(metrics_df)} metrics")
        
        return products_df, metrics_df
    
    def process_data(self, products_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Process raw data into features"""
        logger.info("Processing data...")
        
        processor = DataProcessor(products_df, metrics_df)
        self.features_df = processor.engineer_features()
        self.X, self.y, self.feature_cols = processor.prepare_for_training(self.features_df)
        
        processor.save_processed_data(self.features_df)
        
        logger.info(f"Data processed. Features: {self.X.shape}, Target: {self.y.shape}")
    
    def train_all_models(self):
        """Train all three models"""
        logger.info("Starting model training...")
        
        # Train XGBoost (faster, better baseline)
        logger.info("\n=== Training XGBoost ===")
        self.xgboost_model = XGBoostViralityPredictor()
        xgb_metrics = self.xgboost_model.train(self.X, self.y, feature_names=self.feature_cols)
        logger.info(f"XGBoost complete: {xgb_metrics}")
        
        # Train LSTM
        logger.info("\n=== Training LSTM ===")
        self.lstm_model = LSTMViralityPredictor()
        try:
            lstm_metrics, _ = self.lstm_model.train(self.X, self.y, epochs=30)
            logger.info(f"LSTM complete: {lstm_metrics}")
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Using XGBoost only.")
        
        # Create ensemble
        logger.info("\n=== Creating Ensemble ===")
        self.ensemble_model = EnsembleViralityPredictor(
            lstm_model=self.lstm_model,
            xgboost_model=self.xgboost_model
        )
        
        logger.info("All models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all models"""
        logger.info("\n=== Model Evaluation ===")
        
        # XGBoost evaluation
        logger.info("\nXGBoost Metrics:")
        xgb_metrics = self.xgboost_model.train(self.X, self.y, feature_names=self.feature_cols)
        for key, value in xgb_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Ensemble evaluation
        if self.ensemble_model:
            logger.info("\nEnsemble Metrics:")
            ensemble_metrics = self.ensemble_model.evaluate(self.X, self.y)
            for key, value in ensemble_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        return xgb_metrics
    
    def get_feature_importance(self):
        """Get feature importance from XGBoost"""
        if self.xgboost_model:
            return self.xgboost_model.get_feature_importance(top_n=15)
        return {}
    
    def predict_viral_products(self, top_n: int = 10):
        """Get top predicted viral products"""
        predictions, _ = self.ensemble_model.predict(self.X)
        
        self.features_df['predicted_virality'] = predictions
        viral_products = self.features_df.nlargest(top_n, 'predicted_virality')[[
            'product_name', 'category', 'predicted_virality', 'virality_score'
        ]]
        
        return viral_products


def train_full_pipeline():
    """Full training pipeline"""
    logger.info("\n" + "="*50)
    logger.info("EarlyTrendBD - Full Training Pipeline")
    logger.info("="*50)
    
    trainer = ModelTrainer(num_products=150, days_history=30)
    
    # Generate data
    products_df, metrics_df = trainer.generate_data()
    
    # Process data
    trainer.process_data(products_df, metrics_df)
    
    # Train models
    trainer.train_all_models()
    
    # Evaluate
    trainer.evaluate_models()
    
    # Feature importance
    logger.info("\nTop Features:")
    importance = trainer.get_feature_importance()
    for feat, imp in importance.items():
        logger.info(f"  {feat}: {imp:.4f}")
    
    # Viral products
    logger.info("\nTop Predicted Viral Products:")
    viral_products = trainer.predict_viral_products(top_n=10)
    logger.info(f"\n{viral_products}")
    
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    
    return trainer


if __name__ == "__main__":
    train_full_pipeline()
