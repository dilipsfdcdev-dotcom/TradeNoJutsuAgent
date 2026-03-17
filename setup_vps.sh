#!/bin/bash
# XAUUSD Agent VPS Setup Script
# Tested on Ubuntu 22.04 LTS / Windows Server 2022
set -euo pipefail

echo "=== XAUUSD Agent VPS Setup ==="

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Install Docker (optional - for docker-compose deployment)
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker "$USER"
    rm get-docker.sh
fi

# Install docker-compose
if ! command -v docker-compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
fi

# Create project directory
PROJECT_DIR="${HOME}/xauusd_agent"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup PostgreSQL database
sudo -u postgres psql -c "CREATE USER agent WITH PASSWORD 'changeme';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE xauusd_agent OWNER agent;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE xauusd_agent TO agent;" 2>/dev/null || true

# Create directories
mkdir -p logs models backtest_data reports

# Setup .env file if not exists
if [ ! -f xauusd_agent/config/.env ]; then
    cp xauusd_agent/config/.env.example xauusd_agent/config/.env
    echo ">>> Edit xauusd_agent/config/.env with your credentials"
fi

# Create systemd service
sudo tee /etc/systemd/system/xauusd-agent.service > /dev/null << 'EOF'
[Unit]
Description=XAUUSD Trading Agent
After=postgresql.service network.target
Wants=postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/xauusd_agent
Environment=PATH=$HOME/xauusd_agent/venv/bin:$PATH
ExecStart=$HOME/xauusd_agent/venv/bin/python -m xauusd_agent.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Substitute actual user home
sudo sed -i "s|\$HOME|${HOME}|g" /etc/systemd/system/xauusd-agent.service
sudo sed -i "s|\$USER|${USER}|g" /etc/systemd/system/xauusd-agent.service

sudo systemctl daemon-reload
sudo systemctl enable xauusd-agent

echo ""
echo "=== Setup Complete ==="
echo "1. Edit xauusd_agent/config/.env with your credentials"
echo "2. Start with: sudo systemctl start xauusd-agent"
echo "3. Check logs: journalctl -u xauusd-agent -f"
echo "4. Or run directly: python -m xauusd_agent.main"
