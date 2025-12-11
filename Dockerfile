# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final
FROM python:3.12-slim

WORKDIR /app

# Install ffmpeg (static)
COPY --from=mwader/static-ffmpeg:6.1.1 /ffmpeg /usr/local/bin/
COPY --from=mwader/static-ffmpeg:6.1.1 /ffprobe /usr/local/bin/

# Copy Python dependencies
COPY --from=builder /install /usr/local

# Copy Application
COPY app ./app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; import os; port = os.getenv('API_PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host $API_HOST --port $API_PORT --workers 1"]
