FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the Streamlit port
EXPOSE 7860

# Run the all-in-one launcher
CMD ["python", "app.py"]
