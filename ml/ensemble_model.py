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
        n = len(X) if len(X.shape) > 1 else 1
        xgb_pred = None
        lstm_pred = None
        
        if self.xgboost_model:
            try:
                xgb_pred, _ = self.xgboost_model.predict(X)
                xgb_pred = np.asarray(xgb_pred).flatten()
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
        
        if self.lstm_model:
            try:
                raw_lstm_pred, _ = self.lstm_model.predict(X)
                raw_lstm_pred = np.asarray(raw_lstm_pred).flatten()
                if len(raw_lstm_pred) < n:
                    # LSTM needs `sequence_length` rows of history before its first
                    # prediction, so it returns fewer rows than XGBoost for a big batch.
                    # Pad the front with its first available prediction so shapes align
                    # instead of silently dropping LSTM's contribution.
                    pad = n - len(raw_lstm_pred)
                    fill_value = raw_lstm_pred[0] if len(raw_lstm_pred) > 0 else 0.5
                    lstm_pred = np.concatenate([np.full(pad, fill_value), raw_lstm_pred])
                else:
                    lstm_pred = raw_lstm_pred[:n]
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Use only the models that actually produced a prediction, renormalizing
        # weights so a missing model doesn't silently cap the final score.
        active_weights = {}
        if xgb_pred is not None:
            active_weights['xgboost'] = self.weights['xgboost']
        if lstm_pred is not None:
            active_weights['lstm'] = self.weights['lstm']
        
        if not active_weights:
            raise ValueError("Ensemble has no usable model to predict with.")
        
        weight_total = sum(active_weights.values())
        predictions = np.zeros(n)
        if xgb_pred is not None:
            predictions += (active_weights['xgboost'] / weight_total) * xgb_pred
        if lstm_pred is not None:
            predictions += (active_weights['lstm'] / weight_total) * lstm_pred
        
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
