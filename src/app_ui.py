# phishing-chatbot/src/app_ui.py

import streamlit as st
import requests
import json

# Set FastAPI backend URL
FASTAPI_SERVICE_URL = "https://3es3jnmruh.eu-central-1.awsapprunner.com/"
FASTAPI_URL = f"{FASTAPI_SERVICE_URL}/check-email/"

print(f"Streamlit UI will connect to FastAPI at: {FASTAPI_URL}")

# Set page config
st.set_page_config(
    page_title="Phishing Detection Chatbot",
    page_icon="📧",
    layout="centered"
)

# UI title and description
st.title("🎣 Phishing Detection Chatbot")
st.markdown("""
Enter the subject and body of an email below to check if it's likely a phishing attempt.
Our AI model will analyze it, and if suspicious, provide an explanation and recommendations.
""")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses a machine learning model to detect potential phishing emails "
    "and leverages a Large Language Model (GPT-4o) to provide explanations."
)

# Email input fields
st.header("✉️ Email Input")
email_subject = st.text_input("Email Subject (Optional)")
email_body = st.text_area("Email Body", height=250, placeholder="Paste the full email body here...")

# Check button
submit_button = st.button("Check Email for Phishing")

# Handle form submission
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
                # Send POST request to FastAPI backend
                response = requests.post(FASTAPI_URL, json=payload)
                response.raise_for_status()

                # Parse JSON response
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
                        st.info(f"Model Confidence (Legitimate): {1 - result['model_confidence']:.2%}")

                # Display LLM explanation
                if result.get("llm_explanation"):
                    st.markdown("---")
                    st.markdown("#### 🤔 Explanation from AI Assistant:")
                    st.markdown(result["llm_explanation"].replace("\\n", "\n\n"))

                # Display LLM recommendations
                if result.get("llm_recommendations"):
                    st.markdown("---")
                    st.markdown("#### 🛡️ Recommendations from AI Assistant:")
                    st.markdown(result["llm_recommendations"].replace("\\n", "\n\n"))

                if result.get("error_message"):
                    st.error(f"An error occurred: {result['error_message']}")

            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Could not connect to the Phishing Detection API.")
                st.error(f"Error details: {e}")
            except json.JSONDecodeError:
                st.error("Could not parse the response from the API.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Built by Your Name/Team | Using FastAPI, Scikit-learn, NLTK, OpenAI & Streamlit")