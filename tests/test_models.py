import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from ml.xgboost_model import XGBoostViralityPredictor
from ml.ensemble_model import EnsembleViralityPredictor
from data.generators import SyntheticDataGenerator
from data.processor import DataProcessor


def test_xgboost_training():
    """Test XGBoost model training"""
    # Generate sample data
    generator = SyntheticDataGenerator(num_products=50)
    products_df, metrics_df = generator.generate_all(days=10)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    
    # Train model
    model = XGBoostViralityPredictor()
    metrics = model.train(X, y, feature_names=feature_cols, epochs=20)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'f1' in metrics
    assert metrics['accuracy'] > 0.5
    print("✅ XGBoost training test passed")


def test_xgboost_prediction():
    """Test XGBoost predictions"""
    generator = SyntheticDataGenerator(num_products=30)
    products_df, metrics_df = generator.generate_all(days=10)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    
    model = XGBoostViralityPredictor()
    model.train(X, y, feature_names=feature_cols, epochs=20)
    
    predictions, binary_pred = model.predict(X[:5])
    
    assert predictions.shape[0] == 5
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert binary_pred.shape[0] == 5
    print("✅ XGBoost prediction test passed")


def test_feature_importance():
    """Test feature importance extraction"""
    generator = SyntheticDataGenerator(num_products=50)
    products_df, metrics_df = generator.generate_all(days=10)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    
    model = XGBoostViralityPredictor()
    model.train(X, y, feature_names=feature_cols, epochs=20)
    
    importance = model.get_feature_importance(top_n=5)
    
    assert len(importance) > 0
    assert len(importance) <= 5
    print(f"✅ Feature importance test passed. Top features: {list(importance.keys())}")


def test_ensemble_model():
    """Test ensemble model"""
    generator = SyntheticDataGenerator(num_products=40)
    products_df, metrics_df = generator.generate_all(days=10)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    
    xgb_model = XGBoostViralityPredictor()
    xgb_model.train(X, y, feature_names=feature_cols, epochs=20)
    
    ensemble = EnsembleViralityPredictor(xgboost_model=xgb_model)
    predictions, binary_pred = ensemble.predict(X[:5])
    
    assert predictions.shape[0] == 5
    assert np.all((predictions >= 0) & (predictions <= 1))
    print("✅ Ensemble model test passed")


if __name__ == "__main__":
    test_xgboost_training()
    test_xgboost_prediction()
    test_feature_importance()
    test_ensemble_model()
    print("\n✅ All ML model tests passed!")
