# phishing-chatbot/src/app_ui.py

import streamlit as st
import requests # To make HTTP requests to your FastAPI backend
import json     # To handle JSON data
import os
# Get the FastAPI backend URL from an environment variable
# Default to localhost:8000 if the environment variable is not set (for local direct running)
# Docker Compose will set FASTAPI_BACKEND_URL to http://backend:8000
FASTAPI_SERVICE_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")
FASTAPI_URL = f"{FASTAPI_SERVICE_URL}/check-email/"

print(f"Streamlit UI will connect to FastAPI at: {FASTAPI_URL}") # For debugging
# --- Page Configuration (Optional but good practice) ---
st.set_page_config(
    page_title="Phishing Detection Chatbot",
    page_icon="📧",
    layout="centered" # Can be "wide" or "centered"
)

# --- UI Elements ---
st.title("🎣 Phishing Detection Chatbot")
st.markdown("""
Enter the subject and body of an email below to check if it's likely a phishing attempt.
Our AI model will analyze it, and if suspicious, provide an explanation and recommendations.
""")

st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses a machine learning model to detect potential phishing emails "
    "and leverages a Large Language Model (GPT-4o) to provide explanations."
)

st.header("✉️ Email Input")

# Input fields for email subject and body
email_subject = st.text_input("Email Subject (Optional)")
email_body = st.text_area("Email Body", height=250, placeholder="Paste the full email body here...")

# Button to trigger the check
submit_button = st.button("Check Email for Phishing")

# --- Logic to call FastAPI and display results ---
if submit_button:
    if not email_body:
        st.error("⚠️ Please enter the email body to check.")
    else:
        with st.spinner("Analyzing email... This might take a moment with the LLM call..."):
            payload = {
                "subject": email_subject,
                "body": email_body
            }

            try:
                # Make the POST request to your FastAPI backend
                response = requests.post(FASTAPI_URL, json=payload)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                # Parse the JSON response from FastAPI
                result = response.json()

                # Display results
                st.subheader("🔍 Analysis Results:")

                if result.get("is_phishing"):
                    st.error("🚨 This email is LIKELY a PHISHING attempt!")
                    if result.get("model_confidence") is not None:
                        st.warning(f"Model Confidence (Phishing): {result['model_confidence']:.2%}")
                else:
                    st.success("✅ This email appears to be LEGITIMATE.")
                    if result.get("model_confidence") is not None:
                        # For legitimate, confidence is 1 - prob_phishing
                        st.info(f"Model Confidence (Legitimate): {1 - result['model_confidence']:.2%}")

                # Display LLM explanation and recommendations if available
                if result.get("llm_explanation"):
                    st.markdown("---")
                    st.markdown("#### 🤔 Explanation from AI Assistant:")
                    # Replace \n with actual newlines for Streamlit markdown
                    st.markdown(result["llm_explanation"].replace("\\n", "\n\n"))

                if result.get("llm_recommendations"):
                    st.markdown("---")
                    st.markdown("#### 🛡️ Recommendations from AI Assistant:")
                    st.markdown(result["llm_recommendations"].replace("\\n", "\n\n"))

                if result.get("error_message"):
                    st.error(f"An error occurred: {result['error_message']}")

            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Could not connect to the Phishing Detection API. Is the FastAPI server running?")
                st.error(f"Error details: {e}")
            except json.JSONDecodeError:
                st.error("Could not parse the response from the API. The API might have returned an unexpected format.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("Built by Your Name/Team | Using FastAPI, Scikit-learn, NLTK, OpenAI & Streamlit")