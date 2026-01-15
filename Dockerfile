FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    libgomp1 \
    libprotobuf17 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages
COPY requirements.txt .

# Upgrade pip and install CPU-only onnxruntime first
RUN pip install --upgrade pip \
 && pip install --no-cache-dir onnxruntime==1.15.1 \
 && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Cloud Run port
ENV PORT=8080

# Use Gunicorn
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app_flask"]
