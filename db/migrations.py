from sqlalchemy import text
from db.base import engine, Base
from db.models import Product, DailyMetrics, Prediction

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

def drop_db():
    """Drop all tables (for testing)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️  Database dropped")

if __name__ == "__main__":
    init_db()
