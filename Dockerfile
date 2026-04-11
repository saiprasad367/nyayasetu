FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Copy requirements and install
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the default HF Space port
EXPOSE 7860

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
