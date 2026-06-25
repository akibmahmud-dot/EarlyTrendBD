import pytest
import numpy as np
from data.generators import SyntheticDataGenerator
from data.processor import DataProcessor


def test_data_generation():
    """Test synthetic data generation"""
    generator = SyntheticDataGenerator(num_products=10)
    products_df, metrics_df = generator.generate_all(days=7)
    
    assert len(products_df) == 10
    assert len(metrics_df) > 0
    assert 'name' in products_df.columns
    assert 'sales' in metrics_df.columns
    print("✅ Data generation test passed")


def test_data_processing():
    """Test data processing pipeline"""
    generator = SyntheticDataGenerator(num_products=20)
    products_df, metrics_df = generator.generate_all(days=10)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    
    assert len(features_df) == 20
    assert 'virality_score' in features_df.columns
    assert 'is_viral' in features_df.columns
    assert features_df['virality_score'].max() <= 1.0
    assert features_df['virality_score'].min() >= 0.0
    print("✅ Data processing test passed")


def test_feature_engineering():
    """Test feature engineering"""
    generator = SyntheticDataGenerator(num_products=15)
    products_df, metrics_df = generator.generate_all(days=15)
    
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    
    assert X.shape[0] == len(features_df)
    assert len(feature_cols) > 0
    assert len(y) == len(features_df)
    assert X.shape[1] == len(feature_cols)
    print("✅ Feature engineering test passed")


if __name__ == "__main__":
    test_data_generation()
    test_data_processing()
    test_feature_engineering()
    print("\n✅ All data pipeline tests passed!")
