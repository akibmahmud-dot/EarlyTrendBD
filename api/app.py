from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from loguru import logger
from api.routes import router as prediction_router

# Configure logging
logger.add("logs/app.log", rotation="500 MB")

app = FastAPI(
    title="EarlyTrendBD API",
    description="ML/AI system for predicting product virality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(prediction_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EarlyTrendBD API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/v1/health",
            "predict": "POST /api/v1/predict",
            "analyze": "GET /api/v1/analyze/{product_name}",
            "viral_products": "GET /api/v1/viral-products",
            "feature_importance": "GET /api/v1/feature-importance",
            "model_metrics": "GET /api/v1/model-metrics"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
