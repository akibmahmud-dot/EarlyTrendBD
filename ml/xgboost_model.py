import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from loguru import logger
import pickle
import os
from datetime import datetime

logger.add("logs/xgboost_model.log")

class XGBoostViralityPredictor:
    """XGBoost model for virality prediction with feature importance"""
    
    def __init__(self, model_path: str = "models/trained_models"):
        self.model = None
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.feature_names = None
        os.makedirs(model_path, exist_ok=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: list = None, 
              epochs: int = 100) -> Dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create and train model
        self.model = xgb.XGBClassifier(
            max_depth=7,
            learning_rate=0.1,
            n_estimators=epochs,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"XGBoost Metrics: {metrics}")
        
        self.save_model()
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, predictions > 0.5
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """Get top important features"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            importance_dict[self.feature_names[i]] = float(importance)
        
        # Sort and get top N
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
    
    def save_model(self):
        """Save model to disk"""
        model_file = os.path.join(self.model_path, f"xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        scaler_file = os.path.join(self.model_path, "xgboost_scaler.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_file: str):
        """Load saved model"""
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        scaler_file = os.path.join(self.model_path, "xgboost_scaler.pkl")
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {model_file}")
