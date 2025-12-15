# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Set workdir
WORKDIR /app

# System dependencies (for scientific stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code and requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app

# Make sure Python can import `app` package
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 8000

# Default command: run Flask app
CMD ["flask", "--app", "app.main", "run", "--host=0.0.0.0", "--port=8000"]
