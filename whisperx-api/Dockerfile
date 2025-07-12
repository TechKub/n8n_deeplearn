FROM astral/uv:python3.10-bookworm-slim

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

# Set uv path for future layers
# ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Clone WhisperX
RUN git clone https://github.com/m-bain/whisperX.git /app/whisperX
WORKDIR /app/whisperX

RUN uv venv && uv pip install -e .[all,dev]

RUN uv pip install uvicorn fastapi

RUN uv pip list

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

