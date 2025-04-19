# Dockerfile

# --- Base Image ---
# Use an official Python runtime as a parent image
# Using a specific version and the 'slim' variant for smaller size
FROM python:3.10-slim

# --- Environment Variables ---
# Prevents Python from writing pyc files to disc (improves performance on some systems)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal (useful for logging)
ENV PYTHONUNBUFFERED 1

# --- Set Working Directory ---
# Create and set the working directory in the container
WORKDIR /app

# --- Install Dependencies ---
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size, --upgrade pip ensures latest pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code and Assets ---
# Copy the source code into the container
COPY ./src ./src

# Copy model and data directories
# IMPORTANT: For large models/data, consider mounting volumes or fetching
#            at runtime instead of copying directly into the image.
#            We copy them here for simplicity in this example.
COPY ./models ./models
COPY ./data ./data 
# If your app needs access to raw data at runtime

# Create logs directory (ensure it exists for the logger)
# RUN mkdir logs # No need if config.py creates it, but explicit can be safer

# --- Expose Port ---
# Expose the port the app runs on
EXPOSE 8000

# --- Command to Run Application ---
# Command to run the Uvicorn server when the container launches
# Use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "src.sentiment_analysis_service.main:app", "--host", "0.0.0.0", "--port", "8000"]