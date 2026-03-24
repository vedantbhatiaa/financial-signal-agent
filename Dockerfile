FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by psycopg2 (PostgreSQL adapter)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first — Docker caches this layer
# so rebuilds are fast as long as requirements.txt hasn't changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create data directories expected by the pipeline
RUN mkdir -p data/raw data/processed data/lineage data/chroma

# Default command — can be overridden in docker-compose.yml
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
