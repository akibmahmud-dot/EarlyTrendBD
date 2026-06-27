import os
import json
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import pandas as pd
from loguru import logger

logger.add("logs/data_generation.log")

class SyntheticDataGenerator:
    """Generate realistic synthetic product data for testing and training"""
    
    CATEGORIES = [
        "electronics", "fashion", "home", "beauty", "sports", 
        "books", "toys", "food", "automotive", "furniture"
    ]
    
    PLATFORMS = ["daraz", "shopify", "instagram", "twitter"]
    
    PRODUCT_NAMES = [
        "Samsung Galaxy S24", "iPhone 15 Pro", "Sony WH-1000XM5",
        "Apple AirPods Max", "DJI Air 3S", "Bose QuietComfort 45",
        "Canon EOS R6", "GoPro Hero 12", "Meta Quest 3",
        "Dyson V15", "Ninja Blender", "Instant Pot Pro",
        "KitchenAid Stand Mixer", "Nespresso Machine", "Philips Hue Lights",
        "Tesla Model Y", "Peloton Bike", "Oculus Quest",
        "MacBook Pro M3", "Microsoft Surface Laptop"
    ]
    
    def __init__(self, num_products: int = 100):
        self.num_products = num_products
        self.products = []
        self.metrics = []
    
    def generate_product(self, index: int) -> Dict:
        """Generate a single product record"""
        return {
            "id": index,
            "name": random.choice(self.PRODUCT_NAMES) + f" ({index})",
            "category": random.choice(self.CATEGORIES),
            "platform": random.choice(self.PLATFORMS),
            "platform_id": f"prod_{index}_{random.randint(1000, 9999)}",
            "description": f"High-quality product with excellent features",
            "initial_price": round(random.uniform(50, 2000), 2),
            "image_url": f"https://via.placeholder.com/300?text=Product{index}",
            "created_at": (datetime.now(timezone.utc) - timedelta(days=random.randint(1, 90))).isoformat()
        }
    
    def generate_metrics_for_product(self, product_id: int, days: int = 30) -> List[Dict]:
        """Generate daily metrics for a product over N days"""
        metrics = []
        base_sales = random.randint(10, 500)
        base_views = base_sales * random.uniform(5, 20)
        
        # Decide if product will be viral
        is_viral = random.random() < 0.3  # 30% chance
        # Margin shrinks for short histories so randint() never gets an empty range
        # (the original fixed margin of 5 crashes whenever days < 11, e.g. days=7 in tests)
        margin = min(5, max(1, days // 3))
        viral_day = random.randint(margin, days - margin - 1) if is_viral and days > margin * 2 else None
        
        for day in range(days):
            # Apply viral boost if applicable
            if viral_day and abs(day - viral_day) < 5:
                multiplier = 5 + (5 - abs(day - viral_day)) * 2  # Peak at viral day
            else:
                multiplier = 1 + (day / days) * 0.5  # Slight growth over time
            
            date = datetime.now(timezone.utc) - timedelta(days=days - day - 1)
            
            metric = {
                "product_id": product_id,
                "date": date.isoformat(),
                "sales": int(base_sales * multiplier * random.uniform(0.8, 1.2)),
                "views": int(base_views * multiplier * random.uniform(0.8, 1.2)),
                "clicks": int(base_views * multiplier * 0.1),
                "social_mentions": int(random.uniform(0, 500) * multiplier),
                "instagram_posts": int(random.uniform(0, 50) * multiplier),
                "twitter_mentions": int(random.uniform(0, 100) * multiplier),
                "tiktok_videos": int(random.uniform(0, 200) * multiplier),
                "sentiment_score": min(1.0, 0.5 + (random.random() * 0.5) + (multiplier / 10)),
                "engagement_rate": min(0.2, 0.02 + (random.random() * 0.1)),
                "conversion_rate": min(0.1, 0.01 + (random.random() * 0.05)),
                "search_volume": int(random.uniform(100, 5000) * multiplier),
                "trend_rank": random.randint(1, 1000) if multiplier > 1 else None
            }
            metrics.append(metric)
        
        return metrics
    
    def generate_all(self, days: int = 30) -> tuple:
        """Generate all products and their metrics"""
        logger.info(f"Generating {self.num_products} products with {days} days of metrics...")
        
        products_df_list = []
        metrics_df_list = []
        
        for i in range(self.num_products):
            product = self.generate_product(i)
            products_df_list.append(product)
            
            metrics = self.generate_metrics_for_product(i, days)
            metrics_df_list.extend(metrics)
        
        products_df = pd.DataFrame(products_df_list)
        metrics_df = pd.DataFrame(metrics_df_list)
        
        logger.info(f"Generated {len(products_df)} products and {len(metrics_df)} metric records")
        
        return products_df, metrics_df
    
    def save_to_csv(self, output_dir: str = "data/raw"):
        """Save generated data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        products_df, metrics_df = self.generate_all(days=30)
        
        products_path = os.path.join(output_dir, "products.csv")
        metrics_path = os.path.join(output_dir, "daily_metrics.csv")
        
        products_df.to_csv(products_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"Data saved to {products_path} and {metrics_path}")
        
        return products_df, metrics_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator(num_products=150)
    products_df, metrics_df = generator.save_to_csv()
    print(f"\n✅ Generated {len(products_df)} products")
    print(f"✅ Generated {len(metrics_df)} metric records")
    print(f"\n📊 Sample Products:")
    print(products_df.head())
    print(f"\n📈 Sample Metrics:")
    print(metrics_df.head())
