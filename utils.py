#!/usr/bin/env python
"""Utility functions for the EarlyTrendBD system"""

import os
import json
from typing import Dict, Any
from datetime import datetime, timezone
from loguru import logger
import numpy as np
import pandas as pd


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "logs",
        "data/raw",
        "data/processed",
        "models/trained_models",
        "models/checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"✅ Created {len(directories)} directories")


def save_metrics(metrics: Dict[str, Any], filename: str = "metrics.json"):
    """Save model metrics to JSON file"""
    filepath = f"models/{filename}"
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {filepath}")


def load_metrics(filename: str = "metrics.json") -> Dict[str, Any]:
    """Load model metrics from JSON file"""
    filepath = f"models/{filename}"
    
    if not os.path.exists(filepath):
        logger.warning(f"Metrics file not found: {filepath}")
        return {}
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics"""
    print("\n" + "="*50)
    print("MODEL METRICS")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    
    print("="*50 + "\n")


def get_model_info() -> Dict[str, Any]:
    """Get information about trained models"""
    model_dir = "models/trained_models"
    
    if not os.path.exists(model_dir):
        return {"status": "No trained models found"}
    
    files = os.listdir(model_dir)
    model_info = {
        "total_files": len(files),
        "files": files,
        "last_modified": None
    }
    
    if files:
        latest_file = max(
            [os.path.join(model_dir, f) for f in files],
            key=os.path.getctime
        )
        model_info["last_modified"] = datetime.fromtimestamp(
            os.path.getctime(latest_file)
        ).isoformat()
    
    return model_info


def cleanup_old_models(keep_latest: int = 3):
    """Remove old model files, keeping only the latest N"""
    model_dir = "models/trained_models"
    
    if not os.path.exists(model_dir):
        return
    
    files = [
        os.path.join(model_dir, f) 
        for f in os.listdir(model_dir)
    ]
    
    if len(files) <= keep_latest:
        return
    
    # Sort by modification time
    files.sort(key=os.path.getctime, reverse=True)
    
    # Remove old files
    for old_file in files[keep_latest:]:
        try:
            os.remove(old_file)
            logger.info(f"Removed old model: {old_file}")
        except Exception as e:
            logger.error(f"Failed to remove {old_file}: {e}")


def validate_prediction_input(data: Dict[str, Any]) -> bool:
    """Validate prediction input"""
    required_fields = [
        "product_name",
        "category",
        "initial_sales",
        "sentiment_score",
        "social_mentions"
    ]
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate ranges
    if not 0 <= data["sentiment_score"] <= 1:
        logger.error("sentiment_score must be between 0 and 1")
        return False
    
    if data["initial_sales"] < 0:
        logger.error("initial_sales must be non-negative")
        return False
    
    return True


def format_prediction_response(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction response"""
    return {
        "product_name": prediction.get("product_name", "Unknown"),
        "virality_score": round(prediction.get("virality_score", 0), 4),
        "confidence": round(prediction.get("confidence", 0), 4),
        "is_viral": bool(prediction.get("is_viral", False)),
        "recommendation": prediction.get("recommendation", "No recommendation"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics"""
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }


def create_performance_report(metrics: Dict[str, float]) -> str:
    """Create a text report from metrics"""
    report = f"""
╔════════════════════════════════════════╗
║       PERFORMANCE REPORT               ║
╠════════════════════════════════════════╣
║ Accuracy:  {metrics.get('accuracy', 0):.4f}                  ║
║ Precision: {metrics.get('precision', 0):.4f}                  ║
║ Recall:    {metrics.get('recall', 0):.4f}                  ║
║ F1-Score:  {metrics.get('f1', 0):.4f}                  ║
║ AUC:       {metrics.get('auc', 0):.4f}                  ║
╚════════════════════════════════════════╝
    """
    return report


if __name__ == "__main__":
    print("EarlyTrendBD Utilities")
    ensure_directories()
    print("Model Info:", get_model_info())
