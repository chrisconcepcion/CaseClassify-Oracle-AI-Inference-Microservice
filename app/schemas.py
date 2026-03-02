from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict 

class CasePayload(BaseModel):
    """
    The Contract: This defines exactly what the Rails APP Must send to the Oracle
    If Rails send a 'case_id' as a number, this API will reject it 
    automatically with a 422 error.
    """
    case_id: str = Field(..., description="UUID of the case from OCW")
    description: str = Field(..., min_length=20, description="Description of the case")
    client_id: Optional[str] = Field(None, description="The Government/Enterpise Client Id")

    @field_validator("description")
    def check_text_quality(cls, v):
        if "test" in v.lower() and len(v) < 30:
            raise ValueError("Description appears to be a test artifact")
        return v

class PredictionResponse(BaseModel):
    """
    The Promise: This is exactly what we return to Rails.
    """
    model_config = ConfigDict(protected_namespaces=())

    case_id: str
    urgency_score: str #"HIGH", "MEDIUM", "LOW"
    confidence: float
    processing_time_ms: float
    model_version: str 
