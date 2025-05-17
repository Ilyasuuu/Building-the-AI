# Dockerfile

FROM python:3.12-slim AS builder

WORKDIR /app

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV NLTK_DOWNLOAD_DIR=/tmp/nltk_resources_builder
RUN mkdir -p $NLTK_DOWNLOAD_DIR && \
    echo "Downloading NLTK resources..." && \
    /app/venv/bin/python -m nltk.downloader -d $NLTK_DOWNLOAD_DIR stopwords punkt punkt_tab && \
    echo "NLTK resources downloaded."

FROM python:3.12-slim AS final

WORKDIR /app

COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p $NLTK_DATA

COPY --from=builder /tmp/nltk_resources_builder/corpora $NLTK_DATA/corpora
COPY --from=builder /tmp/nltk_resources_builder/tokenizers $NLTK_DATA/tokenizers

RUN /app/venv/bin/python -c "import nltk; nltk.data.find('tokenizers/punkt'); nltk.data.find('corpora/stopwords'); print('NLTK resources verified in final image.')"

COPY src/ ./src/
COPY Models/ ./Models/

ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]