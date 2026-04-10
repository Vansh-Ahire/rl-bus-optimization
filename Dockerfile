FROM python:3.11-slim

LABEL maintainer="openenv-bus-routing"
LABEL description="OpenEnv-compliant RL bus routing environment with DQN agent"

WORKDIR /app

# Install system deps (none needed beyond what slim provides)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure the app is served on 0.0.0.0 for Spaces
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH="/app"

# Default: run the Gradio dashboard + OpenEnv API for Hugging Face Spaces
EXPOSE 7860
CMD ["python", "server/app.py"]
