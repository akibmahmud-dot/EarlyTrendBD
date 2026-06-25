#!/usr/bin/env python
"""Main entry point for EarlyTrendBD"""

import argparse
import sys
from loguru import logger

logger.add("logs/main.log", rotation="500 MB")


def train_models():
    """Train all ML models"""
    logger.info("Starting model training...")
    from ml.train import train_full_pipeline
    try:
        trainer = train_full_pipeline()
        logger.info("✅ Training complete!")
        return trainer
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def start_api():
    """Start FastAPI server"""
    logger.info("Starting API server...")
    import uvicorn
    try:
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"API failed: {e}")
        sys.exit(1)


def run_tests():
    """Run unit tests"""
    logger.info("Running tests...")
    import pytest
    try:
        result = pytest.main(["tests/", "-v", "--tb=short"])
        if result == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.warning("⚠️ Some tests failed")
        return result
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EarlyTrendBD - Product Virality Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train              # Train models
  python main.py api                # Start API server
  python main.py test               # Run tests
  python main.py --help             # Show this help
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "api", "test"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="EarlyTrendBD v1.0.0"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print(f"""
╔════════════════════════════════════════╗
║       EarlyTrendBD Main Entry          ║
╚════════════════════════════════════════╝
    """)
    
    if args.command == "train":
        train_models()
    elif args.command == "api":
        start_api()
    elif args.command == "test":
        run_tests()


if __name__ == "__main__":
    main()
