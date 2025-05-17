# phishing-chatbot/src/app_ui.py (TEMPORARY SIMPLE VERSION)
import streamlit as st
import os # Just for printing, can be removed if not needed for this test

st.set_page_config(page_title="Simple App Runner Test", layout="centered")

st.title("Streamlit on App Runner - Simple Test")
st.write("If you see this text and the button below, Streamlit is rendering its basic UI.")
st.write(f"Attempting to read env var (should be None locally unless set): {os.getenv('SOME_TEST_VAR', 'Not Set')}")


if st.button("Test Interaction"):
    st.balloons()
    st.success("Button works!")

st.info("Check your browser's developer console for WebSocket errors.")