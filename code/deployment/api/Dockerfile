FROM python:3.11-slim

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY code/deployment/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port
EXPOSE 8000

# Default model path can be overridden at runtime
ENV MODEL_PATH=/app/models/model.pth

# Start the API
CMD ["uvicorn", "code.deployment.api.model:app", "--host", "0.0.0.0", "--port", "8000"]


