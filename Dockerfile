FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies for ONNX Runtime
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

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install packages including onnxruntime
RUN pip install --no-cache-dir onnxruntime==1.15.1 \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Cloud Run expects PORT
ENV PORT=8080

# Use gunicorn with higher timeout for large model
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app_flask"]


