# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY rag-backend/requirements.txt backend-requirements.txt
COPY rag-streamlit/requirements.txt frontend-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend-requirements.txt \
    && pip install --no-cache-dir -r frontend-requirements.txt

# Copy application code
COPY rag-backend /app/backend
COPY rag-streamlit /app/frontend

# Create a non-root user
RUN useradd -m myuser
USER myuser

# Set environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Create a script to run both services
COPY <<EOF /app/start.sh
#!/bin/bash
cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8000 &
cd /app/frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0
EOF

RUN chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 