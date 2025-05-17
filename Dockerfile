# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.12-slim AS builder

WORKDIR /app

# 4. Create a virtual environment within the image
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH" 

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Download NLTK resources to a dedicated directory in the builder stage
ENV NLTK_DOWNLOAD_DIR=/tmp/nltk_resources_builder 
RUN mkdir -p $NLTK_DOWNLOAD_DIR && \
    echo "Downloading NLTK resources: stopwords, punkt, and punkt_tab..." && \
    /app/venv/bin/python -m nltk.downloader -d $NLTK_DOWNLOAD_DIR stopwords punkt punkt_tab && \
    echo "NLTK resources downloaded in builder to $NLTK_DOWNLOAD_DIR. Listing contents:" && \
    echo "--- Listing $NLTK_DOWNLOAD_DIR/tokenizers/ ---" && \
    ls -R $NLTK_DOWNLOAD_DIR/tokenizers/ && \
    echo "--- Listing $NLTK_DOWNLOAD_DIR/corpora/ ---" && \
    ls -R $NLTK_DOWNLOAD_DIR/corpora/

# --- Use a new, smaller stage for the final image ---
FROM python:3.12-slim AS final

WORKDIR /app

COPY --from=builder /app/venv /app/venv

ENV NLTK_DATA=/usr/local/share/nltk_data 
RUN mkdir -p $NLTK_DATA

# Copy the downloaded NLTK resources from the builder stage
# Ensure all relevant subdirectories within NLTK_DOWNLOAD_DIR are copied
COPY --from=builder /tmp/nltk_resources_builder/corpora $NLTK_DATA/corpora
COPY --from=builder /tmp/nltk_resources_builder/tokenizers $NLTK_DATA/tokenizers
# If punkt_tab creates its own top-level folder in NLTK_DOWNLOAD_DIR, you might need:
# COPY --from=builder /tmp/nltk_resources_builder/punkt_tab $NLTK_DATA/punkt_tab # (Unlikely, usually part of tokenizers)

# Verification step (should still use the venv python)
RUN /app/venv/bin/python -c "import nltk; nltk.data.find('tokenizers/punkt'); nltk.data.find('corpora/stopwords'); print('NLTK punkt and stopwords found by venv Python in final image.')"
COPY src/ ./src/
COPY Models/ ./Models/

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" 

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]