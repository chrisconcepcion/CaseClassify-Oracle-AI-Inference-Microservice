# CaseClassify Oracle

## Overview
CaseClassify Oracle is a high-performance Python microservice designed to decouple AI inference from a primary Ruby on Rails monolith. It accepts unstructured legal text (e.g., case descriptions, support tickets) and performs real-time classification to determine urgency and risk levels.

This service is architected to be highly fault-tolerant and is specifically optimized for NVIDIA Blackwell (RTX 50-series) GPUs using CUDA 12.8+ drivers.

## Architecture
* **Framework:** FastAPI (Asynchronous Server Gateway Interface) for handling high-concurrency requests.
* **Inference Engine:** PyTorch with Hugging Face Transformers.
* **Model:** DistilBERT (fine-tuned) for low-latency text classification.
* **Validation:** Pydantic v2 for strict API contract enforcement and schema validation.
* **Infrastructure:** Dockerized with the NVIDIA Container Toolkit for production deployment.

## Prerequisites
* Python 3.11+
* NVIDIA Drivers (550.x or higher recommended for Blackwell architecture)
* Docker & NVIDIA Container Toolkit (for containerized deployment)

## Local Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/case-classify-oracle.git](https://github.com/yourusername/case-classify-oracle.git)
    cd case-classify-oracle
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project uses a specific PyTorch build optimized for CUDA 12.4+ (compatible with Blackwell).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

## Docker Deployment (Production)

The Dockerfile is optimized to use the official NVIDIA CUDA 12.8 base image.

1.  **Build the container:**
    ```bash
    docker build -t oracle:v1 .
    ```

2.  **Run with GPU support:**
    Ensure the NVIDIA Container Toolkit is installed on the host machine.
    ```bash
    docker run --gpus all -p 8000:8000 oracle:v1
    ```

## API Usage

### Health Check
Verifies the service status and GPU visibility.

**Request:**
```bash
curl -X GET http://localhost:8000/health