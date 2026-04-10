FROM python:3.12-slim

WORKDIR /app

# Avoid cache to keep image small
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (including model files)
COPY . .

# Default port (can be overridden by environment variable PORT)
ENV PORT 5000
EXPOSE 5000

# Start the app using python (keeps compatibility with existing app structure)
CMD ["python", "app.py"]
