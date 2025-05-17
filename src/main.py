# phishing-chatbot/src/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # For defining request/response models
import joblib
import re
import nltk # For tokenization and stopwords
from nltk.corpus import stopwords # NLTK's stopwords
# The NLTK resources (stopwords, punkt) should now be reliably provided by the Docker image
# built with the updated Dockerfile. We can remove the runtime download block.

from dotenv import load_dotenv
import os
from openai import OpenAI # For OpenAI API

# --- Configuration & Model Loading ---

# Load environment variables (e.g., OPENAI_API_KEY) from .env file
# This works well when uvicorn is run from the project root.
load_dotenv()
print(f"Current working directory when main.py is loaded: {os.getcwd()}") # For debugging .env loading
if os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key found in environment for FastAPI app.")
else:
    print("WARNING: OpenAI API key NOT found in environment. LLM calls will likely fail.")


# Load the TF-IDF vectorizer and the Logistic Regression model
MODEL_DIR = "Models" # Relative to CWD (project root when running uvicorn src.main:app)
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
model_path = os.path.join(MODEL_DIR, 'log_reg_model.joblib')

tfidf_vectorizer = None
log_reg_model = None
try:
    tfidf_vectorizer = joblib.load(vectorizer_path)
    log_reg_model = joblib.load(model_path)
    print("ML Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Model or Vectorizer file not found. Searched in: {vectorizer_path} and {model_path}")
    print("Ensure files are in the 'Models' directory at the project root and Dockerfile copies them.")
    # In a real app, you might want to prevent FastAPI from starting if models don't load.
except Exception as e:
    print(f"CRITICAL ERROR: An error occurred loading model/vectorizer: {e}")


# Initialize OpenAI client (it will use OPENAI_API_KEY from environment)
openai_client = None
try:
    if os.getenv("OPENAI_API_KEY"): # Only attempt if key is found
        openai_client = OpenAI()
        print("OpenAI client initialized for FastAPI app.")
    else:
        print("OpenAI client NOT initialized because API key was not found.")
except Exception as e:
    print(f"Error initializing OpenAI client for FastAPI app: {e}")


# --- Text Preprocessing Function (Copied and adapted from notebook) ---
# This relies on NLTK resources being available from the Docker image build.
try:
    stop_words_english = set(stopwords.words('english'))
except LookupError as e:
    print(f"CRITICAL NLTK LookupError for stopwords in main.py: {e}. Ensure 'stopwords' are in NLTK_DATA.")
    # Fallback or raise, as preprocessing will fail
    stop_words_english = set() # Empty set, will affect preprocessing quality

def preprocess_text_for_prediction(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    # 2. Remove Punctuation (basic: keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError as e:
        print(f"CRITICAL NLTK LookupError for punkt (for word_tokenize) in main.py: {e}. Ensure 'punkt' is in NLTK_DATA.")
        # Fallback or raise, as preprocessing will fail
        tokens = text.split() # Basic split as a very crude fallback
    # 4. Remove Stop Words
    tokens = [token for token in tokens if token not in stop_words_english]
    # 5. Join back to string
    return " ".join(tokens)


# --- FastAPI App Initialization ---
app = FastAPI(title="Phishing Detection Chatbot API")


# --- Pydantic Models for Request and Response ---
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


# --- API Endpoint ---
@app.post("/check-email/", response_model=PredictionResponse)
async def check_email_phishing(email: EmailInput):
    if not tfidf_vectorizer or not log_reg_model:
        print("ML model or vectorizer not loaded at request time. Check startup logs.")
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
        print(f"ML Model Prediction: {'Phishing' if is_phishing_bool else 'Legitimate'}, Confidence (Phishing): {confidence_phishing:.4f}")
    except Exception as e:
        print(f"Error during ML model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    llm_explanation_text = None
    llm_recommendations_text = None

    if is_phishing_bool:
        if not openai_client:
            print("OpenAI client not initialized, skipping LLM explanation.")
            llm_explanation_text = "LLM explanation unavailable (client not initialized)."
        # No need to check os.getenv("OPENAI_API_KEY") again here if client initialized successfully based on it
        else:
            print("Email flagged as phishing. Contacting LLM...")
            system_prompt = """You are an AI cybersecurity assistant integrated into a phishing detection chatbot.
Your role is to help users understand why an email might be suspicious and what actions they should take.
Structure your response clearly with distinct sections for "Suspicious Elements" and "Recommendations"."""
            user_prompt = f"""An email submitted by the user has been flagged as a **potential phishing attempt** by our machine learning model.
Please analyze the following email content:

-----------------------------------
Subject: {email.subject if email.subject else "N/A"}

Body:
{email.body[:1500]}... 
-----------------------------------

Based on this email, please provide:
1.  A brief, user-friendly explanation of suspicious elements you can identify in the email that are common phishing indicators (e.g., sense of urgency, generic greetings, suspicious links/requests, grammar issues). Be specific to the provided email content if possible.
2.  Clear, actionable security recommendations for the user.
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
                if "Recommendations**" in llm_full_response and "**Suspicious Elements**" in llm_full_response :
                    parts = llm_full_response.split("**Recommendations**", 1)
                    llm_explanation_text = parts[0].replace("**Suspicious Elements**", "").strip()
                    llm_recommendations_text = parts[1].strip() if len(parts) > 1 else "Could not parse recommendations."
                else:
                    llm_explanation_text = llm_full_response
                    llm_recommendations_text = "Recommendations section not clearly found in response."
            except Exception as e:
                print(f"Error during OpenAI API call from FastAPI: {e}")
                llm_explanation_text = f"LLM explanation unavailable due to API error: {e}"

    return PredictionResponse(
        status_code=200,
        is_phishing=is_phishing_bool,
        model_confidence=confidence_phishing,
        llm_explanation=llm_explanation_text,
        llm_recommendations=llm_recommendations_text
    )

@app.get("/")
async def root():
    return {"message": "Phishing Detection Chatbot API is running. Use the /check-email/ endpoint to analyze emails."}

# To run this app (from the project root directory 'phishing-chatbot/'):
# uvicorn src.main:app --reload