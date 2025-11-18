FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System libs needed by TensorFlow and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy server and model files
COPY server.py /app/server.py
COPY models /app/models

# Install Python dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    pillow \
    numpy

EXPOSE 8000

CMD ["python", "server.py"]
