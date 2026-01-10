FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Cloud Run will provide PORT
ENV PORT=8080

# Use Gunicorn to run the Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 300 app:app_flask







