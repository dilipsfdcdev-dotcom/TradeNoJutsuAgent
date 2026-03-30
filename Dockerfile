FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code
COPY agent/ ./agent/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# Expose ports
EXPOSE 8000 8765

# Run the agent
CMD ["python", "-m", "agent.main"]
