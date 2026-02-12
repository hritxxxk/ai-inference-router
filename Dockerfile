# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models_cache

# Set work directory
WORKDIR /app

# Install system dependencies for ChromaDB and other libraries
RUN apt-get update && apt-get install -y 
    build-essential 
    curl 
    software-properties-common 
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model to the image for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db

# Expose port
EXPOSE 8000

# Run the application using gunicorn with uvicorn workers for production
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
