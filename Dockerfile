FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY xauusd_agent/ xauusd_agent/

# Create directories
RUN mkdir -p logs models backtest_data reports

# Run
CMD ["python", "-m", "xauusd_agent.main"]
