# Use stable Python version
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "10000"]