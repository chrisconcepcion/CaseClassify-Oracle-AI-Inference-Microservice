# Bring Celery to handle background work as it's the standard library
# for this purpose.
from celery import Celery

# Connect our Oracle and the Refinery 
from app.main import oracle_brain
from data_pipeline.processor import get_reproducible_pipeline 

# Time Management. We need to collect metrics. 
import time

# Initialize Celery to use Redis as a Queue. 
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task(name="process_legal_batch")
def process_legal_batch(batch_id: int):
    """
    Requirement A: Backpressure. 
    This function runs in the background and not processed as the request comes in.
    """
    
    start_time = time.time()

    # 1. Use the Refinery (Project 2)
    pipeline = get_reproducible_pipeline()
    clean_features = pipeline.fit_transform(raw_cases)

    # 2. Grab the Oracle 
    # We process in a loop/batch to safely handle backpressure.
    results = []
    for featurevector in clean_features:
        prediction = oracle_brain.predict(featurevector)
        results.append(prediction)

    duration = time.time - start_time 
    return {"status": "completed", "count": len(results), "duration": duration}
