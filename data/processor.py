import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger
import os
from datetime import datetime, timedelta

logger.add("logs/data_processing.log")

class DataProcessor:
    """Process and engineer features for ML models"""
    
    def __init__(self, products_df: pd.DataFrame, metrics_df: pd.DataFrame):
        self.products_df = products_df.copy()
        self.metrics_df = metrics_df.copy()
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
    
    def engineer_features(self) -> pd.DataFrame:
        """Create advanced features for virality prediction"""
        logger.info("Engineering features...")
        
        # Convert date to datetime
        self.metrics_df['date'] = pd.to_datetime(self.metrics_df['date'])
        
        # Merge product info with metrics
        data = self.metrics_df.merge(
            self.products_df, left_on='product_id', right_on='id', how='left'
        )
        
        # Group by product for time-series features
        grouped = data.groupby('product_id')
        
        features_list = []
        
        for product_id, group in grouped:
            group = group.sort_values('date')
            
            # Aggregate features
            agg_features = {
                'product_id': product_id,
                'product_name': group['name'].iloc[0] if 'name' in group.columns else 'Unknown',
                'category': group['category'].iloc[0] if 'category' in group.columns else 'unknown',
                
                # Sales features
                'avg_daily_sales': group['sales'].mean(),
                'max_daily_sales': group['sales'].max(),
                'sales_std': group['sales'].std(),
                'sales_trend': (group['sales'].iloc[-1] - group['sales'].iloc[0]) / (group['sales'].iloc[0] + 1),
                
                # View features
                'avg_daily_views': group['views'].mean(),
                'max_daily_views': group['views'].max(),
                'views_trend': (group['views'].iloc[-1] - group['views'].iloc[0]) / (group['views'].iloc[0] + 1),
                
                # Engagement features
                'avg_engagement_rate': group['engagement_rate'].mean(),
                'avg_conversion_rate': group['conversion_rate'].mean(),
                'avg_sentiment_score': group['sentiment_score'].mean(),
                
                # Social media features
                'total_social_mentions': group['social_mentions'].sum(),
                'total_instagram_posts': group['instagram_posts'].sum(),
                'total_twitter_mentions': group['twitter_mentions'].sum(),
                'total_tiktok_videos': group['tiktok_videos'].sum(),
                'total_social_engagement': (group['instagram_posts'].sum() + 
                                          group['twitter_mentions'].sum() + 
                                          group['tiktok_videos'].sum()),
                
                # Volatility (indicator of trends)
                'sales_volatility': group['sales'].rolling(3, min_periods=1).std().mean(),
                'view_volatility': group['views'].rolling(3, min_periods=1).std().mean(),
                
                # Search volume
                'avg_search_volume': group['search_volume'].mean(),
                'peak_search_volume': group['search_volume'].max(),
                
                # Peak indicators
                'has_trend_rank': (group['trend_rank'].notna().sum() > 0).astype(int),
                'days_in_trend': group['trend_rank'].notna().sum(),
            }
            
            features_list.append(agg_features)
        
        features_df = pd.DataFrame(features_list)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        # Calculate virality label using NORMALIZED components (0-1 range each),
        # then label the top 30% as viral via percentile threshold.
        # (Raw sums like total_social_engagement scale with `days`, so a fixed
        # divisor like /1000 broke down once products had >10 days of history.)
        social_raw = features_df['total_social_engagement']
        social_norm = (social_raw - social_raw.min()) / (social_raw.max() - social_raw.min() + 1e-9)
        
        growth_raw = features_df['sales_trend'].clip(lower=0)
        growth_norm = (growth_raw - growth_raw.min()) / (growth_raw.max() - growth_raw.min() + 1e-9)
        
        sentiment_norm = features_df['avg_sentiment_score']  # already 0-1
        
        virality_indicator = social_norm * 0.4 + growth_norm * 0.4 + sentiment_norm * 0.2
        threshold = virality_indicator.quantile(0.70)  # top 30% -> viral
        
        features_df['virality_score'] = virality_indicator
        features_df['is_viral'] = (virality_indicator > threshold).astype(int)
        
        logger.info(f"Engineered {len(features_df)} feature sets")
        logger.info(f"Viral products: {features_df['is_viral'].sum()} / {len(features_df)}")
        
        return features_df
    
    def prepare_for_training(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML model training"""
        logger.info("Preparing data for training...")
        
        # Select feature columns (exclude identifiers and targets)
        feature_cols = [
            'avg_daily_sales', 'max_daily_sales', 'sales_std', 'sales_trend',
            'avg_daily_views', 'max_daily_views', 'views_trend',
            'avg_engagement_rate', 'avg_conversion_rate', 'avg_sentiment_score',
            'total_social_mentions', 'total_instagram_posts', 'total_twitter_mentions',
            'total_tiktok_videos', 'total_social_engagement',
            'sales_volatility', 'view_volatility',
            'avg_search_volume', 'peak_search_volume',
            'has_trend_rank', 'days_in_trend'
        ]
        
        X = features_df[feature_cols].values
        y = features_df['is_viral'].values
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        logger.info(f"Training data shape: {X_scaled.shape}")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
        
        return X_scaled, y, feature_cols
    
    def save_processed_data(self, features_df: pd.DataFrame, output_dir: str = "data/processed"):
        """Save processed features to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "features.csv")
        features_df.to_csv(output_path, index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Generate sample data
    from data.generators import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(num_products=150)
    products_df, metrics_df = generator.generate_all(days=30)
    
    # Process data
    processor = DataProcessor(products_df, metrics_df)
    features_df = processor.engineer_features()
    X, y, feature_cols = processor.prepare_for_training(features_df)
    processor.save_processed_data(features_df)
    
    print(f"\n✅ Data Processing Complete!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {feature_cols}")
