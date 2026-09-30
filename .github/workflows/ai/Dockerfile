FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all AI files
COPY . .

# Create required directories
RUN mkdir -p trained_models data

EXPOSE 5000

CMD ["python", "app.py"]
