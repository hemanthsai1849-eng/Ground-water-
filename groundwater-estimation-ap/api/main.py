"""
Phase 4: API & Visualization Logic
FastAPI application for groundwater level predictions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Optional
import joblib
import os

from schemas import (
    VillageDataRequest,
    PredictionResponse,
    AnomalyResponse,
    HealthCheckResponse,
    ErrorResponse
)

app = FastAPI(
    title="Groundwater Estimation API",
    description="API for predicting groundwater levels in Andhra Pradesh",
    version="1.0.0"
)

# Global variables for model and data
MODEL = None
VILLAGE_DATA = None
HISTORICAL_AVERAGES = None


@app.on_event("startup")
async def startup_event():
    """Load model and data on startup."""
    global MODEL, VILLAGE_DATA, HISTORICAL_AVERAGES
    
    try:
        model_path = os.getenv("MODEL_PATH", "./models/groundwater_model.pkl")
        data_path = os.getenv("DATA_PATH", "./data/processed/village_features.csv")
        
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
        
        if os.path.exists(data_path):
            VILLAGE_DATA = pd.read_csv(data_path)
            HISTORICAL_AVERAGES = VILLAGE_DATA.groupby('village_id')['water_level'].agg(['mean', 'std']).to_dict()
    
    except Exception as e:
        print(f"Warning: Could not load model or data: {e}")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check API health and model status.
    
    Returns:
        HealthCheckResponse with status and model availability
    """
    return HealthCheckResponse(
        status="operational" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        timestamp=datetime.now()
    )


@app.post("/get_village_data/{village_id}", response_model=PredictionResponse)
async def get_village_data(village_id: int):
    """
    Get predicted groundwater level for a specific village.
    
    Args:
        village_id: Unique village identifier
    
    Returns:
        PredictionResponse with predicted level, trend, and alert status
    
    Raises:
        HTTPException: If village not found or prediction fails
    """
    try:
        # Validate village_id
        if village_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="village_id must be a positive integer"
            )
        
        # Check if village exists in data
        if VILLAGE_DATA is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Village data not loaded"
            )
        
        village_records = VILLAGE_DATA[VILLAGE_DATA['village_id'] == village_id]
        
        if village_records.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Village ID {village_id} not found in database"
            )
        
        # Prepare features for prediction
        village_features = village_records.drop(['village_id', 'water_level'], axis=1, errors='ignore').iloc[0]
        
        if MODEL is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Make prediction
        predicted_level = MODEL.predict([village_features.values])[0]
        
        # Get historical average
        historical_avg = VILLAGE_DATA[VILLAGE_DATA['village_id'] == village_id]['water_level'].mean()
        
        # Determine trend
        if predicted_level > historical_avg:
            trend = "RISING"
        elif predicted_level < historical_avg:
            trend = "FALLING"
        else:
            trend = "STABLE"
        
        # Determine alert status
        historical_std = VILLAGE_DATA[VILLAGE_DATA['village_id'] == village_id]['water_level'].std()
        deviation = abs(predicted_level - historical_avg) / (historical_std + 1e-6)
        
        if deviation > 2.0:
            alert_status = "CRITICAL"
        elif deviation > 1.0:
            alert_status = "WARNING"
        else:
            alert_status = "NORMAL"
        
        # Confidence score based on model prediction
        confidence = min(abs(predicted_level) / (abs(predicted_level) + 1), 1.0)
        
        return PredictionResponse(
            village_id=village_id,
            predicted_level=float(predicted_level),
            trend=trend,
            alert_status=alert_status,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(village_ids: list[int]):
    """
    Get predictions for multiple villages at once.
    
    Args:
        village_ids: List of village IDs
    
    Returns:
        List of PredictionResponse objects
    """
    results = []
    
    for village_id in village_ids:
        try:
            response = await get_village_data(village_id)
            results.append(response)
        except HTTPException as e:
            # Continue with next village on error
            continue
    
    return results


@app.post("/anomalies/{village_id}")
async def get_village_anomalies(village_id: int, last_n_days: int = 30):
    """
    Get anomaly detections for a specific village.
    
    Args:
        village_id: Unique village identifier
        last_n_days: Number of days to lookback (default: 30)
    
    Returns:
        List of AnomalyResponse objects
    
    Raises:
        HTTPException: If village not found or no anomalies data available
    """
    if VILLAGE_DATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Village data not loaded"
        )
    
    village_records = VILLAGE_DATA[VILLAGE_DATA['village_id'] == village_id]
    
    if village_records.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Village ID {village_id} not found"
        )
    
    # Mock anomaly detection (in production, this would call the anomaly detection module)
    anomalies = []
    
    return anomalies


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "INVALID_INPUT",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
