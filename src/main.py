# phishing-chatbot/src/main.py
from fastapi.middleware.cors import CORSMiddleware  # Required for CORS handling

app = FastAPI(title="Phishing Detection Chatbot API")

# Configure allowed origins for CORS
origins = [
    "http://localhost",  # Local testing
    "http://localhost:8501",  # Streamlit default port
    "https://imfxqh6ysv.eu-central-1.awsapprunner.com/"  # Frontend URL
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
print(f"Current working directory: {os.getcwd()}")  # Debugging line
if os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key found.")
else:
    print("WARNING: OpenAI API key NOT found.")

# Define paths for ML model and vectorizer
MODEL_DIR = "Models"
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
model_path = os.path.join(MODEL_DIR, 'log_reg_model.joblib')

# Load ML model and TF-IDF vectorizer
tfidf_vectorizer = None
log_reg_model = None
try:
    tfidf_vectorizer = joblib.load(vectorizer_path)
    log_reg_model = joblib.load(model_path)
    print("ML Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Model or Vectorizer not found in {vectorizer_path} and {model_path}")
except Exception as e:
    print(f"CRITICAL ERROR loading model/vectorizer: {e}")

# Initialize OpenAI client if API key is available
openai_client = None
try:
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI()
        print("OpenAI client initialized.")
    else:
        print("OpenAI client NOT initialized (missing API key).")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

# Load NLTK stopwords for text preprocessing
try:
    stop_words_english = set(stopwords.words('english'))
except LookupError as e:
    print(f"NLTK LookupError for stopwords: {e}")
    stop_words_english = set()  # Fallback to empty set

def preprocess_text_for_prediction(text: str) -> str:
    """Preprocess text for model prediction."""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    try:
        tokens = nltk.word_tokenize(text)  # Tokenize
    except LookupError as e:
        print(f"NLTK LookupError for punkt tokenizer: {e}")
        tokens = text.split()  # Fallback tokenization
    tokens = [token for token in tokens if token not in stop_words_english]  # Remove stopwords
    return " ".join(tokens)  # Rejoin tokens

# Create FastAPI app instance
app = FastAPI(title="Phishing Detection Chatbot API")

# Define request/response models
class EmailInput(BaseModel):
    subject: str | None = ""
    body: str

class PredictionResponse(BaseModel):
    status_code: int
    is_phishing: bool
    model_confidence: float | None = None
    llm_explanation: str | None = None
    llm_recommendations: str | None = None
    error_message: str | None = None

# Phishing detection endpoint
@app.post("/check-email/", response_model=PredictionResponse)
async def check_email_phishing(email: EmailInput):
    if not tfidf_vectorizer or not log_reg_model:
        print("ML model or vectorizer not loaded at request time.")
        raise HTTPException(status_code=503, detail="ML model or vectorizer not loaded. Service unavailable.")

    print(f"Received email for checking. Subject: {email.subject[:50] if email.subject else 'N/A'}...")

    combined_text = (email.subject if email.subject else "") + " " + email.body
    processed_text = preprocess_text_for_prediction(combined_text)
    print(f"Processed text (first 100 chars): {processed_text[:100]}...")

    try:
        text_features = tfidf_vectorizer.transform([processed_text])
    except Exception as e:
        print(f"Error during TF-IDF transformation: {e}")
        raise HTTPException(status_code=500, detail=f"Error transforming text: {e}")

    try:
        prediction = log_reg_model.predict(text_features)[0]
        probabilities = log_reg_model.predict_proba(text_features)[0]
        confidence_phishing = float(probabilities[1])
        is_phishing_bool = bool(prediction == 1)
        print(f"ML Model Prediction: {'Phishing' if is_phishing_bool else 'Legitimate'}, Confidence: {confidence_phishing:.4f}")
    except Exception as e:
        print(f"Error during ML model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    llm_explanation_text = None
    llm_recommendations_text = None

    if is_phishing_bool:
        if not openai_client:
            print("OpenAI client not initialized, skipping LLM explanation.")
            llm_explanation_text = "LLM explanation unavailable (client not initialized)."
        else:
            print("Email flagged as phishing. Contacting LLM...")
            system_prompt = """You are an AI cybersecurity assistant integrated into a phishing detection chatbot.
Your role is to help users understand why an email might be suspicious and what actions they should take.
Structure your response clearly with distinct sections for "Suspicious Elements" and "Recommendations"."""  # System prompt for LLM

            user_prompt = f"""An email submitted by the user has been flagged as a **potential phishing attempt** by our machine learning model.
Please analyze the following email content:

-----------------------------------
Subject: {email.subject if email.subject else "N/A"}

Body:
{email.body[:1500]}... 
-----------------------------------

Based on this email, please provide:
1. A brief, user-friendly explanation of suspicious elements you can identify in the email that are common phishing indicators.
2. Clear, actionable security recommendations for the user.
"""
            messages_for_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                completion = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_for_llm,
                    temperature=0.7,
                    max_tokens=700
                )
                llm_full_response = completion.choices[0].message.content
                print("LLM response received.")
                if "Recommendations**" in llm_full_response and "**Suspicious Elements**" in llm_full_response:
                    parts = llm_full_response.split("**Recommendations**", 1)
                    llm_explanation_text = parts[0].replace("**Suspicious Elements**", "").strip()
                    llm_recommendations_text = parts[1].strip() if len(parts) > 1 else "Could not parse recommendations."
                else:
                    llm_explanation_text = llm_full_response
                    llm_recommendations_text = "Recommendations section not clearly found in response."
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                llm_explanation_text = f"LLM explanation unavailable due to API error: {e}"

    return PredictionResponse(
        status_code=200,
        is_phishing=is_phishing_bool,
        model_confidence=confidence_phishing,
        llm_explanation=llm_explanation_text,
        llm_recommendations=llm_recommendations_text
    )

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Phishing Detection Chatbot API is running. Use /check-email/ to analyze emails."}

# To run: uvicorn src.main:app --reload