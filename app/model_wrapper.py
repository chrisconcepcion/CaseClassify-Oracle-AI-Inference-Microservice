import torch 
import time 
from transformers import pipeline 

class AIModelWrapper:
    """
    Decouples ML Logic from API Layer.
    Ensures the model is loaded into GPU VRAM once.
    """
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        print(f"--- Initializing Oracle on {self.device} ---")

        # Load the transformer pipeline
        # Using a distilled model ensures low-latancy for "RealTime" Oracles
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device
        )

    def process_case(self, text: str):
        """
        Runs infernence and maps 'Sentiment' to 'Urgency' for OCW
        """
        start_time = time.time()

        # Standard NLP Inference
        results = self.classifier(text)[0]

        # Domain Logic: Map Negative Sentiment to "HIGH" Urgency
        # This mirros a "Sore Spot" fix identified in support tickets
        urgency = "HIGH" if results['label'] == 'NEGATIVE' else "LOW"

        duration = (time.time() - start_time) * 1000 

        return {
            "urgency": urgency,
            "confidence": round(results['score'], 4),
            "latency_ms": round(duration, 2)
        }

# Singleton instance: Loaded during app startup
oracle_brain = AIModelWrapper()
