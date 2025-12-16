# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

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

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Make sure Python can import `app` package
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 8000

# Development stage
FROM base AS development
COPY . /app
CMD ["python", "-m", "flask", "--app", "app.main:app", "run", "--host=0.0.0.0", "--port=8000", "--reload", "--debug"]

# Production stage
FROM base AS production
COPY . /app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
