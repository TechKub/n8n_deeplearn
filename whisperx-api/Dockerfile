FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    unzip \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using official script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set uv path for future layers
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Clone WhisperX
RUN git clone https://github.com/m-bain/whisperX.git /app/whisperX
WORKDIR /app/whisperX

# IMPORTANT: explicitly reference PATH in same layer
RUN PATH="/root/.cargo/bin:$PATH" && \
    uv venv && \
    uv pip install -e .[all,dev]

EXPOSE 8000

CMD ["uv", "pip", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
