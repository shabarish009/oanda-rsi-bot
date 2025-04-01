# Use official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot script
COPY oanda_rsi_bot.py .

# Default command to run your trading bot
CMD ["python", "oanda_rsi_bot.py"]
