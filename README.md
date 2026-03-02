CaseClassify-Oracle: AI Inference Microservice

Overview

This repository contains a high-performance Python microservice designed to act as a RealTime Oracle for the OpenCaseWare ecosystem. OpenCaseWare is a metadata-driven B2B SaaS platform serving over 50 government and enterprise clients, including the NFTA and Mattel.

As the Lead Architect, I designed this service to solve specific "sore spots" identified through Support Ticket analysis:

    Decoupled Intelligence: By isolating the Transformer-based inference from the Ruby on Rails monolith, we prevent ML memory spikes from impacting core web availability.

    Highly Fault Tolerant: Implements a strict API Contract via Pydantic to ensure data integrity across the distributed system.

    Hardware Optimized: Configured for NVIDIA Blackwell (RTX 50-series) architecture using CUDA 12.8 to minimize inference latency for real-time legal document processing.

AI Capabilities

    Transformer Implementation: Utilizes a distilled BERT architecture for low-latency text classification.

    Feature Engineering: Includes a pandas and scikit-learn pipeline to transform recursive SCTL (Single Case Type Linkage) metadata into flat feature vectors for risk prediction.

    Pre-processing: Automated entity extraction from Case Datum attachments to reduce manual data entry for government clerks.

Tech Stack

    Language: Python 3.11+ 

    Framework: FastAPI (Asynchronous API) 

    ML Engine: PyTorch (CUDA 12.8 Optimized) 

    Models: Hugging Face Transformers 

    Infrastructure: Docker, Redis (Task Queuing)# CaseClassify-Oracle-AI-Inference-Microservice
