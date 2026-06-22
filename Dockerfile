FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit with optimized settings
CMD ["streamlit", "run", "models/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--logger.level=error"]
