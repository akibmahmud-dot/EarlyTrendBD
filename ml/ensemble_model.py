import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger
import warnings

logger.add("logs/ensemble_model.log")

class EnsembleViralityPredictor:
    """Ensemble model combining LSTM and XGBoost predictions"""
    
    def __init__(self, lstm_model=None, xgboost_model=None):
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.weights = {'lstm': 0.4, 'xgboost': 0.6}  # XGBoost gets more weight
    
    def set_weights(self, lstm_weight: float = 0.4, xgboost_weight: float = 0.6):
        """Adjust ensemble weights"""
        total = lstm_weight + xgboost_weight
        self.weights = {
            'lstm': lstm_weight / total,
            'xgboost': xgboost_weight / total
        }
        logger.info(f"Ensemble weights updated: {self.weights}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        predictions = np.zeros(len(X) if len(X.shape) > 1 else 1)
        
        if self.xgboost_model:
            try:
                xgb_pred, _ = self.xgboost_model.predict(X)
                predictions += self.weights['xgboost'] * xgb_pred
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
        
        if self.lstm_model:
            try:
                lstm_pred, _ = self.lstm_model.predict(X)
                if isinstance(lstm_pred, np.ndarray):
                    lstm_pred = lstm_pred.flatten()
                predictions += self.weights['lstm'] * lstm_pred[:len(predictions)]
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # If only one model available, use its predictions
        if predictions.sum() == 0:
            if self.xgboost_model:
                predictions, _ = self.xgboost_model.predict(X)
            elif self.lstm_model:
                predictions, _ = self.lstm_model.predict(X)
        
        return predictions, predictions > 0.5
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        predictions, binary_predictions = self.predict(X)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = {
                'accuracy': accuracy_score(y, binary_predictions.astype(int)),
                'precision': precision_score(y, binary_predictions.astype(int), zero_division=0),
                'recall': recall_score(y, binary_predictions.astype(int), zero_division=0),
                'f1': f1_score(y, binary_predictions.astype(int), zero_division=0),
            }
            
            try:
                metrics['auc'] = roc_auc_score(y, predictions)
            except:
                metrics['auc'] = 0.0
        
        logger.info(f"Ensemble Metrics: {metrics}")
        return metrics
