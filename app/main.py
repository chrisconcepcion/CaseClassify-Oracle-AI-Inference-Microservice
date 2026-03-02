from fastapi import FastAPI, HTTPException 
from app.schemas import CasePayload, PredictionResponse
from app.model_wrapper import oracle_brain

app = FastAPI(title="OCW AI Oracle")

@app.get("/health")
def health():
    # Crucial for 'High Fault Tolerant Systems' 
    return {"status": "online", "gpu_connected": oracle_brain.device == "cuda:0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_urgency(payload: CasePayload):
    """
    The main endpoint for the Rails monolith to hit.
    """
    try:
        # Pass the valiated description to our AI logic
        prediction = oracle_brain.process_case(payload.description)

        return PredictionResponse(
            case_id=payload.case_id,
            urgency_score=prediction["urgency"],
            confidence=prediction["confidence"],
            processing_time_ms=prediction["latency_ms"],
            model_version=oracle_brain.model_name
        )
    except Exception as e:
        # Log and handle exceptions
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Oracle Failure")