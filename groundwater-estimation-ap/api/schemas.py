"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class VillageDataRequest(BaseModel):
    """Schema for village data request."""
    village_id: int = Field(..., description="Unique village identifier")


class PredictionResponse(BaseModel):
    """Schema for groundwater prediction response."""
    village_id: int
    predicted_level: float = Field(..., description="Predicted groundwater level in meters")
    trend: str = Field(..., description="Trend status: RISING, FALLING, or STABLE")
    alert_status: str = Field(..., description="Alert level: NORMAL, WARNING, or CRITICAL")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    timestamp: datetime


class AnomalyResponse(BaseModel):
    """Schema for anomaly detection response."""
    village_id: int
    date: datetime
    water_level: float
    anomaly_score: float
    status: str = Field(..., description="Status: CRITICAL_ANOMALY or NORMAL")
    rolling_std: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Schema for API health check."""
    status: str = Field(..., description="API status: operational or degraded")
    model_loaded: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error_code: str
    message: str
    timestamp: datetime
    village_id: Optional[int] = None
