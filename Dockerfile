FROM python:3.10-slim

# Create non-root user with UID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install system build dependencies and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as root so packages are available system-wide
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project with proper user permissions
COPY --chown=user:user . .

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Expose the Streamlit port
EXPOSE 7860

# Run the all-in-one supervisor launcher
CMD ["python", "app.py"]
