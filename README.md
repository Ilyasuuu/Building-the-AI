# AI-Powered Phishing Detection Chatbot

## Overview
A production-ready chatbot that detects phishing attempts in real-time by analyzing user-submitted text (e.g., emails, messages) using NLP and machine learning. Provides actionable security recommendations via prompt-engineered LLM responses. Deployed on AWS with a CI/CD pipeline for scalability and efficiency.

## Features
- Classifies text as phishing or legitimate using supervised ML models.
- Generates user-friendly explanations and recommendations via Grok 3 or Claude 3.7.
- Optimized for low latency and high accuracy.
- Containerized with Docker and deployed on AWS with GitHub Actions.

## Tech Stack
- **ML/NLP**: Python, Scikit-learn, PyTorch, Pandas, NumPy, NLTK
- **LLM**: Open AI, 4o
- **Backend**: FastAPI
- **Deployment**: Docker, AWS ECS/Lambda, GitHub Actions
- **Version Control**: Git/GitHub