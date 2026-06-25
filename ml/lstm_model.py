import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger
import pickle
import os
from datetime import datetime

logger.add("logs/lstm_model.log")

class LSTMViralityPredictor:
    """LSTM-based time series model for virality prediction"""
    
    def __init__(self, sequence_length: int = 7, model_path: str = "models/trained_models"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        os.makedirs(model_path, exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM architecture"""
        logger.info(f"Building LSTM model with input shape {input_shape}")
        
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert data into sequences for LSTM"""
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        logger.info("Preparing data for LSTM training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        if len(X_seq) < 10:
            logger.warning(f"Insufficient data for sequences: {len(X_seq)}. Using fallback approach.")
            X_seq = X_scaled.reshape(-1, 1, X_scaled.shape[1])
            y_seq = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Build and train model
        self.model = self.build_model(X_train.shape[1:])
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"LSTM Metrics: {metrics}")
        
        self.save_model()
        
        return metrics, history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if len(X_scaled.shape) == 1:
            X_scaled = X_scaled.reshape(1, -1)
        
        if X_scaled.shape[0] < self.sequence_length:
            X_scaled = X_scaled.reshape(-1, 1, X_scaled.shape[1]) if len(X_scaled.shape) == 2 else X_scaled
        else:
            X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
            X_scaled = X_seq
        
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions, predictions > 0.5
    
    def save_model(self):
        """Save model to disk"""
        model_file = os.path.join(self.model_path, f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
        scaler_file = os.path.join(self.model_path, "lstm_scaler.pkl")
        
        self.model.save(model_file)
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_file: str):
        """Load saved model"""
        self.model = keras.models.load_model(model_file)
        scaler_file = os.path.join(self.model_path, "lstm_scaler.pkl")
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Model loaded from {model_file}")
