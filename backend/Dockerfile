FROM python:3.11-slim

WORKDIR /app

# Prevent mutex/threading issues
ENV USE_TORCH=1
ENV TOKENIZERS_PARALLELISM=false

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
